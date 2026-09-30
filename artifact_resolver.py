"""Phase 2: resolve URLs without download, then fetch code and budgeted data."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx2
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from artifact import ensure_schema, get_experiment
from environment import score_repository

USER_AGENT = "repro-artifact-resolver/1.0 (+https://github.com; research crawler)"
DEFAULT_STORE = Path("artifacts/stage1")
MAX_REPO_BYTES = 5 * 1024 * 1024 * 1024
DATA_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".parquet",
    ".feather",
    ".xlsx",
    ".xls",
    ".sav",
    ".sas7bdat",
    ".dta",
    ".sqlite",
    ".db",
    ".h5",
    ".hdf5",
    ".npz",
    ".npy",
    ".rds",
    ".rda",
    ".pkl",
    ".pickle",
}
CODE_EXTENSIONS = {
    ".py",
    ".r",
    ".rmd",
    ".ipynb",
    ".js",
    ".ts",
    ".jl",
    ".java",
    ".cpp",
    ".c",
    ".go",
    ".rs",
    ".m",
    ".scala",
}
PAPER_HOST_HINTS = ("doi.org", "acm.org", "arxiv.org", "springer", "wiley")
PERSONAL_HINTS = ("github.io", "gitlab.io", "wordpress", "blogspot", "sites.google", "wixsite")
INSTITUTIONAL_SUFFIXES = (".edu", ".ac.uk", ".ac.", ".edu.")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def request_url(value: str) -> str | None:
    """Return a safe absolute HTTP(S) URL for a verbatim extracted URL.

    The claim-extraction stage retains URLs exactly as written, so valid web
    addresses often omit ``https://``. Preserve that original value in SQLite,
    but add HTTPS only for a bare host/path. Relative paths and unsupported
    schemes cannot be resolved without a base URL.
    """
    # urlparse silently discards embedded tabs and newlines, so a URL broken across
    # lines in the PDF parses as valid while the raw string is rejected by httpx.
    # Drop them here so the returned URL is the one that was actually parsed.
    raw = value.strip().translate({0x09: None, 0x0A: None, 0x0D: None})
    if not raw or raw.startswith(("/", "#", "?")):
        return None
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return raw
    if "://" in raw or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw):
        return None
    candidate = "https://" + raw.lstrip("/")
    parsed = urlparse(candidate)
    host = parsed.hostname
    if not parsed.netloc or not host or any(char.isspace() for char in host):
        return None
    return candidate


def classify_host(host: str | None, reachable: bool) -> str:
    if not reachable or not host:
        return "unreachable"
    host = host.lower().rstrip(".")
    if host in {"github.com", "www.github.com", "gist.github.com"} or host.endswith(".github.com"):
        return "github"
    if "gitlab" in host:
        return "gitlab"
    if host in {"osf.io", "www.osf.io"} or host.endswith(".osf.io"):
        return "osf"
    if "zenodo" in host:
        return "zenodo"
    if "figshare" in host:
        return "figshare"
    if "dataverse" in host:
        return "dataverse"
    if any(hint in host for hint in PERSONAL_HINTS):
        return "personal"
    if host.endswith(".edu") or ".ac." in host or host.endswith(".ac.uk"):
        return "institutional"
    return "other"


def classify_kind(url: str, content_type: str | None, host_class: str) -> str:
    lowered = url.lower()
    ctype = (content_type or "").lower()
    if any(hint in lowered for hint in PAPER_HOST_HINTS) or "pdf" in ctype:
        paper = True
    else:
        paper = False
    code = host_class in {"github", "gitlab"} or any(
        lowered.endswith(ext) for ext in CODE_EXTENSIONS
    )
    data = any(token in lowered for token in ("/data", "dataset", "csv", "zenodo", "figshare", "dataverse"))
    if host_class in {"zenodo", "figshare", "dataverse", "osf"}:
        data = True
    if paper and not code and not data:
        return "paper"
    if code and data:
        return "both"
    if code:
        return "code"
    if data:
        return "data"
    return "unknown"


class RobotsCache:
    def __init__(self, client: httpx2.AsyncClient) -> None:
        self.client = client
        self._cache: dict[str, RobotFileParser | None] = {}

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._cache:
            robots_url = origin + "/robots.txt"
            parser = RobotFileParser()
            try:
                response = await self.client.get(robots_url, timeout=15.0, follow_redirects=True)
                if response.status_code >= 400:
                    self._cache[origin] = None
                else:
                    parser.parse(response.text.splitlines())
                    self._cache[origin] = parser
            except httpx2.HTTPError:
                self._cache[origin] = None
        parser = self._cache[origin]
        if parser is None:
            return True
        return parser.can_fetch(USER_AGENT, url)


async def _sleep_backoff(attempt: int) -> None:
    await asyncio.sleep(min(2**attempt, 30))


async def fetch_with_policy(
    client: httpx2.AsyncClient,
    method: str,
    url: str,
    *,
    robots: RobotsCache,
    follow: bool = False,
) -> httpx2.Response:
    if not await robots.allowed(url):
        raise PermissionError("robots_disallowed")
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = await client.request(
                method, url, follow_redirects=follow, timeout=30.0
            )
            if response.status_code == 429 or response.status_code >= 500:
                last_error = httpx2.HTTPStatusError(
                    f"HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
                await _sleep_backoff(attempt)
                continue
            return response
        except httpx2.HTTPError as exc:
            last_error = exc
            await _sleep_backoff(attempt)
    assert last_error is not None
    raise last_error


async def resolve_url(
    client: httpx2.AsyncClient, robots: RobotsCache, url: str
) -> dict[str, Any]:
    retrieved_at = _now()
    chain: list[dict[str, Any]] = []
    current = request_url(url)
    if current is None:
        return {
            "ok": False,
            "retrieval_state": "error",
            "error_text": "invalid_or_relative_url",
            "redirect_chain": chain,
            "retrieved_at": retrieved_at,
            "host_class": "unreachable",
        }
    final_response: httpx2.Response | None = None
    try:
        for _ in range(10):
            response = await fetch_with_policy(client, "HEAD", current, robots=robots)
            location = response.headers.get("location")
            chain.append(
                {"url": str(response.url), "status": response.status_code, "location": location}
            )
            if response.status_code in {401, 403}:
                final_response = response
                break
            if response.is_redirect and location:
                current = str(response.next_request.url) if response.next_request else urljoin(str(response.url), location)
                continue
            if response.status_code == 405:
                response = await fetch_with_policy(client, "GET", current, robots=robots)
                chain.append({"url": str(response.url), "status": response.status_code, "location": None})
            final_response = response
            break
        else:
            return {
                "ok": False,
                "retrieval_state": "error",
                "error_text": "redirect_loop",
                "redirect_chain": chain,
                "retrieved_at": retrieved_at,
            }
    except PermissionError:
        return {
            "ok": False,
            "retrieval_state": "robots_disallowed",
            "error_text": "robots.txt disallowed this URL",
            "redirect_chain": chain,
            "retrieved_at": retrieved_at,
            "host_class": "other",
        }
    except (httpx2.HTTPError, httpx2.InvalidURL, ValueError) as exc:
        return {
            "ok": False,
            "retrieval_state": "unreachable",
            "error_text": f"{type(exc).__name__}: {exc}",
            "redirect_chain": chain,
            "retrieved_at": retrieved_at,
            "host_class": "unreachable",
        }
    assert final_response is not None
    status = final_response.status_code
    final_url = str(final_response.url)
    host = urlparse(final_url).hostname
    auth_walled = status in {401, 403} or _looks_like_login(final_url, final_response)
    state = "auth_walled" if auth_walled else ("resolved" if status < 400 else "unreachable")
    host_class = classify_host(host, state != "unreachable")
    content_type = final_response.headers.get("content-type")
    size = _content_length(final_response)
    metadata: dict[str, Any] = {}
    size_source = "content-length" if size is not None else None
    if host_class in {"github", "gitlab", "osf", "zenodo", "figshare", "dataverse"} and not auth_walled:
        extra = await repository_metadata(client, robots, final_url, host_class)
        metadata.update(extra)
        if extra.get("size_bytes") is not None:
            size = extra["size_bytes"]
            size_source = extra.get("size_source", "api")
    return {
        "ok": True,
        "status_code": status,
        "redirect_chain": chain,
        "final_url": final_url,
        "final_host": host,
        "host_class": host_class,
        "content_type": content_type,
        "artifact_kind": classify_kind(final_url, content_type, host_class),
        "retrieval_state": state,
        "size_bytes": size,
        "size_source": size_source,
        "retrieved_at": retrieved_at,
        "metadata": metadata,
        "error_text": None if state != "unreachable" else f"HTTP {status}",
    }


def _looks_like_login(url: str, response: httpx2.Response) -> bool:
    lowered = url.lower()
    if any(token in lowered for token in ("/login", "/signin", "/sso", "auth0", "shibboleth")):
        return True
    return bool(response.headers.get("www-authenticate"))


def _content_length(response: httpx2.Response) -> int | None:
    value = response.headers.get("content-length")
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


async def repository_metadata(
    client: httpx2.AsyncClient, robots: RobotsCache, url: str, host_class: str
) -> dict[str, Any]:
    try:
        if host_class == "github":
            return await _github_metadata(client, robots, url)
        if host_class == "gitlab":
            return await _gitlab_metadata(client, robots, url)
        if host_class == "zenodo":
            return await _zenodo_metadata(client, robots, url)
        if host_class == "osf":
            return await _osf_metadata(client, robots, url)
        if host_class == "figshare":
            return await _figshare_metadata(client, robots, url)
        if host_class == "dataverse":
            return await _dataverse_metadata(client, robots, url)
    except Exception as exc:
        return {"api_error": f"{type(exc).__name__}: {exc}"}
    return {}


def _github_repo_slug(url: str) -> tuple[str, str] | None:
    path = urlparse(url).path.strip("/").split("/")
    if len(path) < 2:
        return None
    return path[0], path[1].removesuffix(".git")


async def _github_metadata(client: httpx2.AsyncClient, robots: RobotsCache, url: str) -> dict[str, Any]:
    slug = _github_repo_slug(url)
    if not slug:
        return {}
    owner, repo = slug
    api = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = await fetch_with_policy(client, "GET", api, robots=robots, follow=True)
    if response.status_code >= 400:
        return {"api_status": response.status_code}
    data = response.json()
    size_kb = data.get("size")
    return {
        "size_bytes": size_kb * 1024 if isinstance(size_kb, int) else None,
        "size_source": "github_api_size_kb",
        "default_branch": data.get("default_branch"),
        "pushed_at": data.get("pushed_at"),
        "license": (data.get("license") or {}).get("spdx_id"),
        "clone_url": data.get("clone_url"),
        "html_url": data.get("html_url"),
    }


async def _gitlab_metadata(client: httpx2.AsyncClient, robots: RobotsCache, url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    project = parsed.path.strip("/").removesuffix(".git")
    if not project:
        return {}
    encoded = project.replace("/", "%2F")
    api = f"{parsed.scheme}://{parsed.netloc}/api/v4/projects/{encoded}"
    response = await fetch_with_policy(client, "GET", api, robots=robots, follow=True)
    if response.status_code >= 400:
        return {"api_status": response.status_code}
    data = response.json()
    size = (data.get("statistics") or {}).get("repository_size")
    return {
        "size_bytes": size,
        "size_source": "gitlab_api_statistics",
        "http_url_to_repo": data.get("http_url_to_repo"),
        "last_activity_at": data.get("last_activity_at"),
    }


async def _zenodo_metadata(client: httpx2.AsyncClient, robots: RobotsCache, url: str) -> dict[str, Any]:
    match = re.search(r"record(?:s)?/(\d+)", url)
    if not match:
        return {}
    api = f"https://zenodo.org/api/records/{match.group(1)}"
    response = await fetch_with_policy(client, "GET", api, robots=robots, follow=True)
    if response.status_code >= 400:
        return {"api_status": response.status_code}
    data = response.json()
    files = data.get("files") or []
    total = sum(int(item.get("size") or 0) for item in files)
    return {
        "size_bytes": total,
        "size_source": "zenodo_files",
        "files": [
            {"key": item.get("key"), "size": item.get("size"), "links": item.get("links")}
            for item in files
        ],
    }


async def _osf_metadata(client: httpx2.AsyncClient, robots: RobotsCache, url: str) -> dict[str, Any]:
    match = re.search(r"osf\.io/([a-z0-9]+)", url, re.I)
    if not match:
        return {}
    guid = match.group(1)
    api = f"https://api.osf.io/v2/guids/{guid}/"
    response = await fetch_with_policy(client, "GET", api, robots=robots, follow=True)
    if response.status_code >= 400:
        return {"api_status": response.status_code}
    return {"osf_type": (response.json().get("data") or {}).get("type"), "size_source": "osf_guid"}


async def _figshare_metadata(client: httpx2.AsyncClient, robots: RobotsCache, url: str) -> dict[str, Any]:
    match = re.search(r"articles/[^/]+/(\d+)", url)
    if not match:
        return {}
    api = f"https://api.figshare.com/v2/articles/{match.group(1)}"
    response = await fetch_with_policy(client, "GET", api, robots=robots, follow=True)
    if response.status_code >= 400:
        return {"api_status": response.status_code}
    data = response.json()
    files = data.get("files") or []
    total = sum(int(item.get("size") or 0) for item in files)
    return {"size_bytes": total, "size_source": "figshare_files", "files": files}


async def _dataverse_metadata(client: httpx2.AsyncClient, robots: RobotsCache, url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    persistent = None
    match = re.search(r"persistentId=([^&]+)", url)
    if match:
        persistent = match.group(1)
    if not persistent:
        return {}
    api = f"{parsed.scheme}://{parsed.netloc}/api/datasets/:persistentId/?persistentId={persistent}"
    response = await fetch_with_policy(client, "GET", api, robots=robots, follow=True)
    if response.status_code >= 400:
        return {"api_status": response.status_code}
    files = (
        (((response.json().get("data") or {}).get("latestVersion") or {}).get("files")) or []
    )
    total = 0
    for item in files:
        datafile = item.get("dataFile") or {}
        total += int(datafile.get("filesize") or 0)
    return {"size_bytes": total, "size_source": "dataverse_files"}


async def _repository_files(
    client: httpx2.AsyncClient, robots: RobotsCache, url: str, host_class: str | None
) -> list[dict[str, Any]]:
    """Resolve a repository *landing page* to its actual downloadable files.

    B1: the fetcher streamed `final_url` directly, which for a DOI is the landing page. Both
    Dataverse and figshare answer those with **202 Accepted and an empty body**, which passes
    `raise_for_status()`, so a zero-byte file was written and the row marked `fetched`. Nine
    data URLs were lost that way. The size helpers above already knew the API shape; only the
    download path did not use it.

    Returns [] when the host is not a known repository, in which case the caller streams the
    URL as before.
    """
    host = (host_class or "").lower()
    lowered = url.lower()

    if host == "figshare" or "figshare.com" in lowered:
        match = re.search(r"articles/(?:[^/]+/)?(\d+)", url)
        if not match:
            return []
        api = f"https://api.figshare.com/v2/articles/{match.group(1)}"
        response = await fetch_with_policy(client, "GET", api, robots=robots, follow=True)
        if response.status_code >= 400:
            return []
        out = []
        for item in response.json().get("files") or []:
            href = item.get("download_url")
            if href:
                out.append(
                    {
                        "name": item.get("name") or "download.bin",
                        "url": href,
                        "size": int(item.get("size") or 0),
                    }
                )
        return out

    if host == "dataverse" or "dataverse" in lowered:
        parsed = urlparse(url)
        match = re.search(r"persistentId=([^&]+)", url)
        if not match:
            return []
        persistent = match.group(1)
        api = (
            f"{parsed.scheme}://{parsed.netloc}/api/datasets/:persistentId/"
            f"?persistentId={persistent}"
        )
        response = await fetch_with_policy(client, "GET", api, robots=robots, follow=True)
        if response.status_code >= 400:
            return []
        files = (
            ((response.json().get("data") or {}).get("latestVersion") or {}).get("files")
        ) or []
        out = []
        for item in files:
            datafile = item.get("dataFile") or {}
            file_id = datafile.get("id")
            if file_id is None:
                continue
            out.append(
                {
                    "name": datafile.get("filename") or f"{file_id}.bin",
                    "url": f"{parsed.scheme}://{parsed.netloc}/api/access/datafile/{file_id}",
                    "size": int(datafile.get("filesize") or 0),
                }
            )
        return out

    return []


async def _persist_resolution(catalog: Any, row: Any, result: dict[str, Any]) -> None:
    await catalog.db.execute(
        "UPDATE artifact_urls SET status_code=?,redirect_chain_json=?,final_url=?,final_host=?,"
        "host_class=?,content_type=?,artifact_kind=?,retrieval_state=?,size_bytes=?,size_source=?,"
        "retrieved_at=?,error_text=?,metadata_json=? WHERE id=?",
        (
            result.get("status_code"),
            json.dumps(result.get("redirect_chain") or [], ensure_ascii=False),
            result.get("final_url"),
            result.get("final_host"),
            result.get("host_class"),
            result.get("content_type"),
            result.get("artifact_kind"),
            result.get("retrieval_state"),
            result.get("size_bytes"),
            result.get("size_source"),
            result.get("retrieved_at") or _now(),
            result.get("error_text"),
            json.dumps(result.get("metadata") or {}, ensure_ascii=False, sort_keys=True),
            row["id"],
        ),
    )


async def _select_urls(catalog: Any, args: Any) -> list[Any]:
    clauses = ["1=1"]
    params: list[Any] = []
    if getattr(args, "experiment", None):
        experiment = await get_experiment(catalog, args.experiment)
        clauses.append("u.experiment_id=?")
        params.append(experiment["id"])
    if getattr(args, "year", None):
        clauses.append("p.collection_year=?")
        params.append(args.year)
    elif not getattr(args, "all", False) and not getattr(args, "urls_file", None):
        clauses.append("p.collection_year IN (2021,2022)")
    if getattr(args, "paper_id", None):
        clauses.append("u.paper_id=?")
        params.append(args.paper_id)
    if not getattr(args, "force", False):
        clauses.append("u.retrieved_at IS NULL")
    sql = (
        "SELECT u.*,p.collection_year FROM artifact_urls u JOIN papers p ON p.id=u.paper_id WHERE "
        + " AND ".join(clauses)
        + " ORDER BY p.collection_year,u.id"
    )
    cursor = await catalog.db.execute(sql, params)
    return await cursor.fetchall()


async def command_resolve_run(args: Any, logger: Any, Catalog: Any) -> int:
    store_rows: list[dict[str, Any]] = []
    async with Catalog(Path(args.database)) as catalog:
        await ensure_schema(catalog)
        if getattr(args, "urls_file", None):
            rows = []
            for line in Path(args.urls_file).read_text(encoding="utf-8").splitlines():
                url = line.strip()
                if not url or url.startswith("#"):
                    continue
                store_rows.append({"id": None, "original_url": url, "paper_id": None})
        else:
            if not getattr(args, "all", False) and getattr(args, "year", None) is None:
                logger.info(
                    "resolve_default_window",
                    years=[2021, 2022],
                    hint="pass --all to resolve every extracted URL regardless of year",
                )
            rows = await _select_urls(catalog, args)
            store_rows = [dict(row) for row in rows]
        limits: dict[str, asyncio.Semaphore] = defaultdict(lambda: asyncio.Semaphore(1))
        timeout = httpx2.Timeout(30.0)

        async def handle(item: dict[str, Any], client: httpx2.AsyncClient, robots: RobotsCache) -> dict[str, Any]:
            normalized = request_url(item["original_url"])
            host = urlparse(normalized).hostname if normalized else "invalid"
            async with limits[host or "invalid"]:
                return await resolve_url(client, robots, item["original_url"])

        if not store_rows:
            logger.info("resolve_nothing_to_do")
            return 0
        gathered = []
        async with httpx2.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=False,
        ) as client:
            robots = RobotsCache(client)
            for item in tqdm(store_rows, desc="Resolving URLs", unit="url", disable=args.no_progress):
                gathered.append(await handle(item, client, robots))
                if item.get("id") is not None:
                    await _persist_resolution(catalog, item, gathered[-1])
                    await catalog.db.commit()
        if getattr(args, "urls_file", None):
            console = Console()
            table = Table(title="Resolver fixture run")
            table.add_column("url")
            table.add_column("state")
            table.add_column("status")
            table.add_column("host")
            table.add_column("size")
            for item, result in zip(store_rows, gathered, strict=True):
                table.add_row(
                    item["original_url"],
                    str(result.get("retrieval_state")),
                    str(result.get("status_code")),
                    str(result.get("final_host")),
                    str(result.get("size_bytes")),
                )
            console.print(table)
        else:
            await _print_volume_table(catalog)
    logger.info("resolve_complete", urls=len(store_rows))
    return 0


async def _print_volume_table(catalog: Any) -> None:
    cursor = await catalog.db.execute(
        "SELECT p.collection_year year,u.paper_id,p.canonical_doi doi,"
        "SUM(COALESCE(u.size_bytes,0)) bytes,COUNT(*) urls,"
        "SUM(CASE WHEN u.retrieval_state='auth_walled' THEN 1 ELSE 0 END) auth_walled,"
        "SUM(CASE WHEN u.retrieval_state='unreachable' THEN 1 ELSE 0 END) unreachable "
        "FROM artifact_urls u JOIN papers p ON p.id=u.paper_id "
        "WHERE u.retrieved_at IS NOT NULL GROUP BY u.paper_id ORDER BY bytes DESC,u.paper_id"
    )
    rows = await cursor.fetchall()
    console = Console()
    table = Table(title="Phase 2a data volume by paper")
    table.add_column("paper_id")
    table.add_column("year")
    table.add_column("doi")
    table.add_column("urls")
    table.add_column("bytes")
    table.add_column("auth_walled")
    table.add_column("unreachable")
    total = 0
    for row in rows:
        total += int(row["bytes"] or 0)
        table.add_row(
            str(row["paper_id"]),
            str(row["year"]),
            str(row["doi"] or ""),
            str(row["urls"]),
            str(row["bytes"]),
            str(row["auth_walled"]),
            str(row["unreachable"]),
        )
    console.print(table)
    console.print(f"Total measured bytes: {total}")


async def command_resolve_report(args: Any, logger: Any, Catalog: Any) -> int:
    async with Catalog(Path(args.database)) as catalog:
        await ensure_schema(catalog)
        await _print_volume_table(catalog)
    return 0


def _inventory(root: Path) -> dict[str, Any]:
    files: list[str] = []
    languages: dict[str, int] = defaultdict(int)
    readme = False
    license_present = False
    manifests: list[str] = []
    last_commit = None
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith(".git/") or rel == ".git":
            continue
        if path.is_file():
            files.append(rel)
            languages[path.suffix.lower() or path.name] += 1
            name = path.name.lower()
            if name.startswith("readme"):
                readme = True
            if name.startswith("license") or name.startswith("copying"):
                license_present = True
            if name in {
                "requirements.txt",
                "environment.yml",
                "environment.yaml",
                "pyproject.toml",
                "renv.lock",
                "uv.lock",
                "poetry.lock",
                "dockerfile",
            }:
                manifests.append(rel)
    git_dir = root / ".git"
    if git_dir.exists():
        try:
            last_commit = subprocess.check_output(
                ["git", "-C", str(root), "log", "-1", "--format=%cI"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            last_commit = None
    return {
        "file_tree": files[:5000],
        "file_count": len(files),
        "languages": dict(languages),
        "readme": readme,
        "license": license_present,
        "manifests": manifests,
        "last_commit": last_commit,
    }


def _hash_tree(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    files = sorted(path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts)
    for path in files:
        rel = path.relative_to(root).as_posix().encode()
        digest.update(rel)
        data = path.read_bytes()
        total += len(data)
        digest.update(data)
    return digest.hexdigest(), total


def _clone(url: str, dest: Path, timeout: int = 600) -> tuple[str, str | None]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return "fetched", None
    except subprocess.CalledProcessError as exc:
        text = (exc.stderr or exc.stdout or str(exc)).lower()
        if any(token in text for token in ("authentication", "permission denied", "403", "401")):
            return "auth_walled", exc.stderr or str(exc)
        return "error", exc.stderr or str(exc)
    except FileNotFoundError:
        return "error", "git is not installed"
    except subprocess.TimeoutExpired:
        return "error", "git clone timed out"


async def command_fetch_code(args: Any, logger: Any, Catalog: Any) -> int:
    store_root = Path(getattr(args, "store", DEFAULT_STORE))
    max_bytes = int(getattr(args, "max_repo_gb", 5) * 1024 * 1024 * 1024)
    async with Catalog(Path(args.database)) as catalog:
        await ensure_schema(catalog)
        clauses = [
            "u.retrieval_state IN ('resolved','fetched','skipped_too_large','auth_walled')",
            "u.host_class IN ('github','gitlab')",
        ]
        params: list[Any] = []
        if getattr(args, "experiment", None):
            experiment = await get_experiment(catalog, args.experiment)
            clauses.append("u.experiment_id=?")
            params.append(experiment["id"])
        cursor = await catalog.db.execute(
            "SELECT u.* FROM artifact_urls u WHERE "
            + " AND ".join(clauses)
            + " ORDER BY COALESCE(u.size_bytes,0), u.id",
            params,
        )
        rows = await cursor.fetchall()
        for row in tqdm(rows, desc="Fetching code", unit="repo", disable=args.no_progress):
            cursor = await catalog.db.execute(
                "SELECT id FROM artifact_snapshots WHERE url_id=?", (row["id"],)
            )
            if await cursor.fetchone() and not getattr(args, "force", False):
                continue
            size = row["size_bytes"]
            now = _now()
            if size is not None and size > max_bytes:
                await catalog.db.execute(
                    "UPDATE artifact_urls SET retrieval_state='skipped_too_large',retrieved_at=? WHERE id=?",
                    (now, row["id"]),
                )
                await catalog.db.commit()
                continue
            dest = store_root / str(row["paper_id"]) / row["url_sha256"] / "content"
            clone_url = row["final_url"] or row["original_url"]
            metadata = json.loads(row["metadata_json"] or "{}")
            clone_url = metadata.get("clone_url") or metadata.get("http_url_to_repo") or clone_url
            state, error = await asyncio.to_thread(_clone, clone_url, dest)
            if state != "fetched":
                await catalog.db.execute(
                    "UPDATE artifact_urls SET retrieval_state=?,error_text=?,retrieved_at=? WHERE id=?",
                    (state, error, now, row["id"]),
                )
                await catalog.db.commit()
                continue
            tree_hash, tree_size = await asyncio.to_thread(_hash_tree, dest)
            if tree_size > max_bytes:
                shutil.rmtree(dest, ignore_errors=True)
                await catalog.db.execute(
                    "UPDATE artifact_urls SET retrieval_state='skipped_too_large',size_bytes=?,retrieved_at=? WHERE id=?",
                    (tree_size, now, row["id"]),
                )
                await catalog.db.commit()
                continue
            inventory = await asyncio.to_thread(_inventory, dest)
            components = await asyncio.to_thread(score_repository, dest)
            manifest = {
                "source_url": row["original_url"],
                "resolved_url": row["final_url"],
                "host_class": row["host_class"],
                "sha256": tree_hash,
                "retrieved_at": now,
                "size_bytes": tree_size,
                "artifact_kind": "code",
                "truncated": False,
                "inventory": inventory,
            }
            dest.parent.mkdir(parents=True, exist_ok=True)
            (dest.parent / "manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            await catalog.db.execute(
                "UPDATE artifact_urls SET retrieval_state='fetched',size_bytes=?,retrieved_at=?,error_text=NULL WHERE id=?",
                (tree_size, now, row["id"]),
            )
            cursor = await catalog.db.execute(
                "INSERT INTO artifact_snapshots(url_id,paper_id,store_path,sha256,size_bytes,truncated,retrieved_at,manifest_json) "
                "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(url_id) DO UPDATE SET store_path=excluded.store_path,sha256=excluded.sha256,"
                "size_bytes=excluded.size_bytes,truncated=excluded.truncated,retrieved_at=excluded.retrieved_at,manifest_json=excluded.manifest_json "
                "RETURNING id",
                (
                    row["id"],
                    row["paper_id"],
                    str(dest.parent),
                    tree_hash,
                    tree_size,
                    0,
                    now,
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                ),
            )
            snapshot = await cursor.fetchone()
            await catalog.db.execute(
                "INSERT INTO artifact_environment(snapshot_id,paper_id,url_id,manifest_present,versions_pinned,"
                "lockfile_or_container,seeds_set,relative_paths,creates_output_dirs,runtime_declared,e_score,"
                "components_json,scored_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(snapshot_id) DO UPDATE SET manifest_present=excluded.manifest_present,"
                "versions_pinned=excluded.versions_pinned,lockfile_or_container=excluded.lockfile_or_container,"
                "seeds_set=excluded.seeds_set,relative_paths=excluded.relative_paths,"
                "creates_output_dirs=excluded.creates_output_dirs,runtime_declared=excluded.runtime_declared,"
                "e_score=excluded.e_score,components_json=excluded.components_json,scored_at=excluded.scored_at",
                (
                    snapshot[0],
                    row["paper_id"],
                    row["id"],
                    int(components["manifest_present"]),
                    int(components["versions_pinned"]),
                    int(components["lockfile_or_container"]),
                    int(components["seeds_set"]),
                    int(components["relative_paths"]),
                    int(components["creates_output_dirs"]),
                    int(components["runtime_declared"]),
                    components["E"],
                    json.dumps(components, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
            await catalog.db.commit()
    logger.info("fetch_code_complete")
    return 0


def _is_machine_readable(row: Any) -> bool:
    ctype = (row["content_type"] or "").lower()
    url = (row["final_url"] or row["original_url"] or "").lower()
    if any(token in ctype for token in ("csv", "json", "excel", "parquet", "sqlite", "zip", "octet-stream")):
        return True
    return any(url.endswith(ext) or ext in url for ext in DATA_EXTENSIONS)


async def command_fetch_data(args: Any, logger: Any, Catalog: Any) -> int:
    budget = int(args.budget_gb * 1024 * 1024 * 1024)
    max_bytes = int(getattr(args, "max_repo_gb", 5) * 1024 * 1024 * 1024)
    store_root = Path(getattr(args, "store", DEFAULT_STORE))
    cumulative = 0
    async with Catalog(Path(args.database)) as catalog:
        await ensure_schema(catalog)
        # Papers whose code repository has already been retrieved. Only used when
        # --require-code is set: that filter assembles execution inputs, whereas the
        # executable-candidate census needs data fetched for data-only papers too.
        require_code = bool(getattr(args, "require_code", False))
        cursor = await catalog.db.execute(
            "SELECT DISTINCT paper_id FROM artifact_snapshots"
        )
        code_papers = {row["paper_id"] for row in await cursor.fetchall()}
        cursor = await catalog.db.execute(
            "SELECT u.* FROM artifact_urls u WHERE u.retrieval_state IN ('resolved','fetched') "
            "AND u.artifact_kind IN ('data','both','unknown') AND u.host_class NOT IN ('github','gitlab') "
            "ORDER BY COALESCE(u.size_bytes, 1 << 62), u.id"
        )
        rows = await cursor.fetchall()
        async with httpx2.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=httpx2.Timeout(60.0),
            follow_redirects=True,
        ) as client:
            robots = RobotsCache(client)
            for row in tqdm(rows, desc="Fetching data", unit="file", disable=args.no_progress):
                if require_code and row["paper_id"] not in code_papers:
                    continue
                if not _is_machine_readable(row) and row["artifact_kind"] != "data":
                    continue
                cursor = await catalog.db.execute(
                    "SELECT id FROM artifact_snapshots WHERE url_id=?", (row["id"],)
                )
                if await cursor.fetchone() and not getattr(args, "force", False):
                    continue
                now = _now()
                size = row["size_bytes"]
                if size is not None and size > max_bytes:
                    await catalog.db.execute(
                        "UPDATE artifact_urls SET retrieval_state='skipped_too_large',retrieved_at=? WHERE id=?",
                        (now, row["id"]),
                    )
                    await catalog.db.commit()
                    continue
                if size is not None and cumulative + size > budget:
                    await catalog.db.execute(
                        "UPDATE artifact_urls SET retrieval_state='skipped_budget',retrieved_at=?,error_text=? WHERE id=?",
                        (now, f"stop rule: cumulative {cumulative} + {size} exceeds budget {budget}", row["id"]),
                    )
                    await catalog.db.commit()
                    logger.info("fetch_data_budget_stop", cumulative=cumulative, budget=budget)
                    break
                dest_dir = store_root / str(row["paper_id"]) / row["url_sha256"]
                content_dir = dest_dir / "content"
                content_dir.mkdir(parents=True, exist_ok=True)
                url = row["final_url"] or row["original_url"]
                try:
                    if not await robots.allowed(url):
                        raise PermissionError("robots_disallowed")
                    # B1: a DOI resolves to a landing page, which Dataverse and figshare
                    # answer with 202 and an empty body. Ask the repository API for the
                    # actual files first; fall back to streaming the URL itself.
                    targets = await _repository_files(
                        client, robots, url, row["host_class"]
                    )
                    if not targets:
                        targets = [{"name": None, "url": url, "size": None}]
                    hasher = hashlib.sha256()
                    written = 0
                    auth_walled = False
                    for item in targets:
                        async with client.stream("GET", item["url"]) as response:
                            if response.status_code in {401, 403}:
                                auth_walled = True
                                break
                            response.raise_for_status()
                            name = (
                                item["name"]
                                or Path(urlparse(item["url"]).path).name
                                or "download.bin"
                            )
                            target = content_dir / Path(name).name
                            with target.open("wb") as handle:
                                async for chunk in response.aiter_bytes():
                                    written += len(chunk)
                                    if written > max_bytes:
                                        handle.close()
                                        target.unlink(missing_ok=True)
                                        raise MemoryError("skipped_too_large")
                                    if cumulative + written > budget:
                                        handle.close()
                                        target.unlink(missing_ok=True)
                                        raise OverflowError("skipped_budget")
                                    hasher.update(chunk)
                                    handle.write(chunk)
                    if auth_walled:
                        await catalog.db.execute(
                            "UPDATE artifact_urls SET retrieval_state='auth_walled',status_code=?,retrieved_at=? WHERE id=?",
                            (response.status_code, now, row["id"]),
                        )
                        await catalog.db.commit()
                        continue
                    # B1: never record an empty retrieval as success. A 2xx that yields no
                    # bytes is a failure to obtain the artifact, whatever the status line said.
                    if written == 0:
                        # 'error' rather than a new state: the CHECK constraint on
                        # retrieval_state is a closed vocabulary and SQLite cannot alter it in
                        # place. An empty retrieval IS an error; the reason goes in error_text.
                        await catalog.db.execute(
                            "UPDATE artifact_urls SET retrieval_state='error',retrieved_at=?,"
                            "error_text=? WHERE id=?",
                            (
                                now,
                                f"empty_response: {len(targets)} target(s) returned 0 bytes; "
                                f"landing page or empty deposit, not a dataset",
                                row["id"],
                            ),
                        )
                        await catalog.db.commit()
                        logger.warning(
                            "fetch_data_empty", url_id=row["id"], paper_id=row["paper_id"]
                        )
                        continue
                    digest = hasher.hexdigest()
                    cumulative += written
                    manifest = {
                        "source_url": row["original_url"],
                        "resolved_url": row["final_url"],
                        "host_class": row["host_class"],
                        "sha256": digest,
                        "retrieved_at": now,
                        "size_bytes": written,
                        "artifact_kind": "data",
                        "truncated": False,
                    }
                    (dest_dir / "manifest.json").write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True),
                        encoding="utf-8",
                    )
                    await catalog.db.execute(
                        "UPDATE artifact_urls SET retrieval_state='fetched',size_bytes=?,retrieved_at=? WHERE id=?",
                        (written, now, row["id"]),
                    )
                    await catalog.db.execute(
                        "INSERT INTO artifact_snapshots(url_id,paper_id,store_path,sha256,size_bytes,truncated,retrieved_at,manifest_json) "
                        "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(url_id) DO UPDATE SET store_path=excluded.store_path,"
                        "sha256=excluded.sha256,size_bytes=excluded.size_bytes,retrieved_at=excluded.retrieved_at,manifest_json=excluded.manifest_json",
                        (
                            row["id"],
                            row["paper_id"],
                            str(dest_dir),
                            digest,
                            written,
                            0,
                            now,
                            json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                        ),
                    )
                    await catalog.db.commit()
                except MemoryError:
                    await catalog.db.execute(
                        "UPDATE artifact_urls SET retrieval_state='skipped_too_large',retrieved_at=? WHERE id=?",
                        (now, row["id"]),
                    )
                    await catalog.db.commit()
                except OverflowError:
                    await catalog.db.execute(
                        "UPDATE artifact_urls SET retrieval_state='skipped_budget',retrieved_at=?,error_text=? WHERE id=?",
                        (now, f"stop rule at cumulative {cumulative}", row["id"]),
                    )
                    await catalog.db.commit()
                    logger.info("fetch_data_budget_stop", cumulative=cumulative, budget=budget)
                    break
                except PermissionError:
                    await catalog.db.execute(
                        "UPDATE artifact_urls SET retrieval_state='robots_disallowed',retrieved_at=? WHERE id=?",
                        (now, row["id"]),
                    )
                    await catalog.db.commit()
                except Exception as exc:
                    await catalog.db.execute(
                        "UPDATE artifact_urls SET retrieval_state='error',error_text=?,retrieved_at=? WHERE id=?",
                        (f"{type(exc).__name__}: {exc}", now, row["id"]),
                    )
                    await catalog.db.commit()
    logger.info("fetch_data_complete", cumulative_bytes=cumulative, budget_bytes=budget)
    return 0
