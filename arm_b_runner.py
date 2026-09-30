"""Stage 4 Arm B: independent paper-and-data reanalysis, without author code."""
from __future__ import annotations

import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from screening import sha256_text, utc_now
from arm_a_runner import (
    FAULT_PRECEDENCE,
    _modal,
    _rrl,
    _usage_cost,
    _withheld,
    compare_paper,
    leak_check,
    value_needles,
    value_present,
)

EXPERIMENT_KIND = "arm_b_reanalysis"
HARNESS = "inspect_ai"
ARM_B_PAPERS = (34, 69, 244, 321, 408, 482, 491)
ARM_B_SKIPS = {431: "no recomputable value (Stage 3 adjudication)", 468: "R artifact excluded: frozen Python-only Arm A/B sandbox has no R runtime"}

ARM_B_SCHEMA = r"""
CREATE TABLE IF NOT EXISTS arm_b_experiments (
 id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, status TEXT NOT NULL CHECK(status IN ('draft','frozen','closed')),
 claim_experiment_id INTEGER NOT NULL REFERENCES claim_experiments(id) ON DELETE RESTRICT,
 prompt_name TEXT NOT NULL,prompt_version TEXT NOT NULL,system_prompt TEXT NOT NULL,prompt_sha256 TEXT NOT NULL,
 schema_version TEXT NOT NULL,submission_schema TEXT NOT NULL,schema_sha256 TEXT NOT NULL,
 provider TEXT NOT NULL,model TEXT NOT NULL,reasoning_effort TEXT,repeat_count INTEGER NOT NULL CHECK(repeat_count>0),
 message_limit INTEGER,cost_limit_usd REAL,time_limit_seconds INTEGER NOT NULL,command_timeout_seconds INTEGER NOT NULL,
 network_available INTEGER NOT NULL,harness TEXT NOT NULL,harness_version TEXT NOT NULL,image_tag TEXT NOT NULL,
 configuration_json TEXT NOT NULL,created_at TEXT NOT NULL,frozen_at TEXT
) STRICT;
CREATE TABLE IF NOT EXISTS arm_b_runs (
 id INTEGER PRIMARY KEY,experiment_id INTEGER NOT NULL REFERENCES arm_b_experiments(id) ON DELETE RESTRICT,
 paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE RESTRICT,repetition_index INTEGER NOT NULL CHECK(repetition_index>0),
 snapshot_ids_json TEXT NOT NULL,status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed')),
 custom_id TEXT NOT NULL UNIQUE,prompt_sha256 TEXT NOT NULL,user_prompt TEXT NOT NULL,leak_check TEXT NOT NULL,
 log_path TEXT,eval_status TEXT,limit_hit TEXT,computed INTEGER,produced_value REAL,produced_value_units TEXT,
 produced_metric_name TEXT,produced_value_location TEXT,produced_ranking_json TEXT,produced_group_count INTEGER,
 operationalisation TEXT,assumptions_json TEXT,analysis_script TEXT,could_not_compute_reason TEXT,agent_fault_class TEXT,
 agent_reasoning TEXT,submission_json TEXT,submission_error TEXT,contradictions_json TEXT,evidence_source TEXT,
 evidence_text TEXT,location_verified INTEGER,location_verify_note TEXT,setup_json TEXT,wall_clock_seconds REAL,
 working_seconds REAL,message_count INTEGER,input_tokens INTEGER,output_tokens INTEGER,estimated_cost_usd REAL,
 error_type TEXT,error_text TEXT,created_at TEXT NOT NULL,finished_at TEXT,
 UNIQUE(experiment_id,paper_id,repetition_index)
) STRICT;
CREATE TABLE IF NOT EXISTS arm_b_decisions (
 id INTEGER PRIMARY KEY,experiment_id INTEGER NOT NULL REFERENCES arm_b_experiments(id) ON DELETE RESTRICT,
 paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE RESTRICT,claim_type TEXT,computed INTEGER,
 computed_votes_json TEXT NOT NULL,fault_class TEXT,fault_votes_json TEXT NOT NULL,produced_value REAL,
 produced_value_spread REAL,produced_values_json TEXT NOT NULL,reported_value REAL,reported_units TEXT,metric_name TEXT,
 scale_note TEXT,abs_delta REAL,rel_delta REAL,comparator TEXT NOT NULL,tolerance_verdict TEXT NOT NULL,
 qualitative_verdict TEXT NOT NULL,rrl_level TEXT NOT NULL,comparison_note TEXT,location_verified INTEGER,
 value_disagreement INTEGER NOT NULL DEFAULT 0,needs_adjudication INTEGER NOT NULL DEFAULT 0,
 completed_run_count INTEGER NOT NULL,wall_clock_seconds REAL,aggregated_at TEXT NOT NULL,
 UNIQUE(experiment_id,paper_id)
) STRICT;
"""


async def ensure_arm_b_schema(catalog: Any) -> None:
    for statement in ARM_B_SCHEMA.split(";"):
        if statement.strip():
            await catalog.db.execute(statement)
    await catalog.db.commit()


async def get_experiment(catalog: Any, name: str) -> Any:
    row = await (await catalog.db.execute("SELECT * FROM arm_b_experiments WHERE name=?", (name,))).fetchone()
    if row is None:
        raise ValueError(f"Unknown Arm B experiment: {name}")
    return row


async def _claim_experiment(catalog: Any, name: str | None) -> Any:
    if name:
        cursor = await catalog.db.execute("SELECT id,name,model FROM claim_experiments WHERE name=?", (name,))
    else:
        cursor = await catalog.db.execute("SELECT id,name,model FROM claim_experiments WHERE status='frozen' ORDER BY id DESC LIMIT 1")
    row = await cursor.fetchone()
    if row is None:
        raise ValueError("No frozen Stage 3 claim experiment is available for Arm B")
    return row


async def _candidate_papers(catalog: Any, experiment: Any, paper_id: int | None) -> list[int]:
    requested = (paper_id,) if paper_id is not None else ARM_B_PAPERS
    if paper_id is not None and paper_id not in ARM_B_PAPERS:
        reason = ARM_B_SKIPS.get(paper_id, "not in the seven-paper Arm B scope")
        raise ValueError(f"Paper {paper_id} is not eligible for Arm B: {reason}")
    placeholders = ",".join("?" * len(requested))
    cursor = await catalog.db.execute(
        "SELECT DISTINCT d.paper_id FROM claim_decisions d JOIN artifact_snapshots s ON s.paper_id=d.paper_id "
        "JOIN artifact_urls u ON u.id=s.url_id AND u.artifact_kind IN ('code','data','both') "
        f"WHERE d.experiment_id=? AND d.paper_id IN ({placeholders}) AND s.size_bytes>0 ORDER BY d.paper_id",
        (experiment["claim_experiment_id"], *requested),
    )
    found = [int(row["paper_id"]) for row in await cursor.fetchall()]
    missing = sorted(set(requested) - set(found))
    if missing:
        raise ValueError(f"Arm B data snapshot missing for scoped paper(s): {missing}")
    return found


async def command_arm_b_create(args: Any, logger: Any, Catalog: Any) -> int:
    from arm_b.dataset import DEFAULT_IMAGE_TAG
    from arm_b.prompt import ARM_B_SYSTEM_PROMPT, PROMPT_NAME, PROMPT_VERSION, SCHEMA_VERSION, SUBMISSION_SCHEMA
    import importlib.metadata
    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_b_schema(catalog)
        claim = await _claim_experiment(catalog, getattr(args, "claim_experiment", None))
        config = {"arm": "B", "question": "can the paper and data recover the claim independently of shipped code", "author_code_mounted": False, "paper_mounted": True, "paper_target_value_redacted": True, "reported_value_withheld": True, "comparison": "lakdb arm-b compare outside sandbox", "unspecified_by_paper": "all material for Arm B", "scope_papers": list(ARM_B_PAPERS), "skipped_papers": ARM_B_SKIPS}
        now = utc_now()
        cursor = await catalog.db.execute(
            "INSERT INTO arm_b_experiments(name,status,claim_experiment_id,prompt_name,prompt_version,system_prompt,prompt_sha256,schema_version,submission_schema,schema_sha256,provider,model,reasoning_effort,repeat_count,message_limit,cost_limit_usd,time_limit_seconds,command_timeout_seconds,network_available,harness,harness_version,image_tag,configuration_json,created_at,frozen_at) VALUES (?,'frozen',?,?,?,?,?,?,?,?,'openai',?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (args.name, claim["id"], PROMPT_NAME, PROMPT_VERSION, ARM_B_SYSTEM_PROMPT, sha256_text(ARM_B_SYSTEM_PROMPT), SCHEMA_VERSION, SUBMISSION_SCHEMA, sha256_text(SUBMISSION_SCHEMA), args.model, args.reasoning_effort, args.repeats, args.message_limit, args.cost_limit, args.time_limit, args.command_timeout, int(not args.no_network), HARNESS, importlib.metadata.version("inspect_ai"), DEFAULT_IMAGE_TAG, json.dumps(config, sort_keys=True), now, now),
        )
        await catalog.db.commit()
        logger.info("arm_b_experiment_created", name=args.name, experiment_id=cursor.lastrowid, papers=list(ARM_B_PAPERS), repeats=args.repeats)
    return 0


async def command_arm_b_run(args: Any, logger: Any, Catalog: Any) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required (environment or .env)")
    if getattr(args, "no_progress", False):
        os.environ.setdefault("INSPECT_DISPLAY", "plain")
    from inspect_ai import eval_async
    from arm_b.dataset import build_sample
    from arm_b.task import arm_b as arm_b_task
    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_b_schema(catalog)
        experiment = await get_experiment(catalog, args.experiment)
        cursor = await catalog.db.execute("SELECT name FROM claim_experiments WHERE id=?", (experiment["claim_experiment_id"],))
        claim_name = (await cursor.fetchone())["name"]
        papers = await _candidate_papers(catalog, experiment, getattr(args, "paper_id", None))
        repeats = int(getattr(args, "repeats", None) or experiment["repeat_count"])
        log_root = Path(getattr(args, "log_dir", None) or "artifacts/arm_b/logs")
        ok = failed = skipped = 0
        for paper_id in papers:
            sample = build_sample(args.database, paper_id, claim_name, image_tag=experiment["image_tag"], network=bool(experiment["network_available"]))
            prompt = str(sample.input)
            withheld = await _withheld(catalog, experiment["claim_experiment_id"], paper_id)
            hits = leak_check(prompt + experiment["system_prompt"], value_needles(withheld.get("point_estimate"), withheld.get("verbatim")))
            if hits and not getattr(args, "skip_leak_check", False):
                raise ValueError(f"Paper {paper_id}: target value digits reach the Arm B prompt: {hits}")
            leak_state = "clean" if not hits else f"acknowledged:{','.join(hits)}"
            for repetition in range(1, repeats + 1):
                custom_id = f"armb-{experiment['id']}-{paper_id}-{repetition}"
                existing = await (await catalog.db.execute("SELECT status FROM arm_b_runs WHERE custom_id=?", (custom_id,))).fetchone()
                if existing and existing["status"] == "succeeded" and not getattr(args, "force", False):
                    skipped += 1; continue
                await catalog.db.execute("INSERT INTO arm_b_runs(experiment_id,paper_id,repetition_index,snapshot_ids_json,status,custom_id,prompt_sha256,user_prompt,leak_check,created_at) VALUES (?,?,?,?,'running',?,?,?,?,?) ON CONFLICT(custom_id) DO UPDATE SET status='running',prompt_sha256=excluded.prompt_sha256,user_prompt=excluded.user_prompt,leak_check=excluded.leak_check", (experiment["id"],paper_id,repetition,json.dumps((sample.metadata or {}).get("snapshot_ids", [])),custom_id,sha256_text(prompt),prompt,leak_state,utc_now()))
                await catalog.db.commit()
                try:
                    logs = await eval_async(arm_b_task(paper_id=paper_id,database=str(args.database),claim_experiment=claim_name,reasoning_effort=experiment["reasoning_effort"] or "medium",message_limit=experiment["message_limit"],cost_limit=experiment["cost_limit_usd"],time_limit=experiment["time_limit_seconds"],command_timeout=experiment["command_timeout_seconds"],image_tag=experiment["image_tag"],network=bool(experiment["network_available"])),model=f"openai/{experiment['model']}",log_dir=str(log_root / args.experiment / str(paper_id) / f"rep{repetition}"),fail_on_error=False,score=False)
                except Exception as exc:  # harness failure is a stored run outcome
                    await catalog.db.execute("UPDATE arm_b_runs SET status='failed',error_type=?,error_text=?,finished_at=? WHERE custom_id=?", (type(exc).__name__,str(exc)[:4000],utc_now(),custom_id)); await catalog.db.commit(); failed += 1; continue
                log = logs[0]; status = "succeeded" if log.status == "success" else "failed"
                await catalog.db.execute("UPDATE arm_b_runs SET status=?,log_path=?,eval_status=?,error_type=?,error_text=?,finished_at=? WHERE custom_id=?", (status,log.location,log.status,type(log.error).__name__ if log.error else None,log.error.message if log.error else None,utc_now(),custom_id)); await catalog.db.commit()
                ok += int(status == "succeeded"); failed += int(status != "succeeded")
        logger.info("arm_b_run_complete", experiment=args.experiment, succeeded=ok, failed=failed, skipped=skipped)
    return 0


async def command_arm_b_collect(args: Any, logger: Any, Catalog: Any) -> int:
    from inspect_ai.log import read_eval_log
    from arm_b.submission import parse_submission
    from arm_b.task import EVIDENCE_KEY, SETUP_KEY
    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_b_schema(catalog); experiment = await get_experiment(catalog, args.experiment)
        rows = await (await catalog.db.execute("SELECT custom_id,log_path FROM arm_b_runs WHERE experiment_id=? ORDER BY paper_id,repetition_index", (experiment["id"],))).fetchall()
        parsed = missing = 0
        for row in rows:
            if not row["log_path"]: missing += 1; continue
            try: log = read_eval_log(row["log_path"]); sample = log.samples[0]
            except Exception: missing += 1; continue
            submission, raw, error = parse_submission(sample.output.completion if sample.output else None)
            evidence = (sample.metadata or {}).get(EVIDENCE_KEY) or (sample.store or {}).get(EVIDENCE_KEY) or {}
            setup = (sample.metadata or {}).get(SETUP_KEY) or (sample.store or {}).get(SETUP_KEY) or {}
            limit_hit = sample.limit.type if sample.limit else None
            if submission:
                fields = {"computed": None if submission.computed is None else int(submission.computed), "produced_value": submission.produced_value, "produced_value_units": submission.produced_value_units, "produced_metric_name": submission.produced_metric_name, "produced_value_location": submission.produced_value_location, "produced_ranking_json": json.dumps(submission.produced_ranking) if submission.produced_ranking is not None else None, "produced_group_count": submission.produced_group_count, "operationalisation": submission.operationalisation, "assumptions_json": json.dumps(submission.assumptions_made), "analysis_script": submission.analysis_script, "could_not_compute_reason": submission.could_not_compute_reason, "agent_fault_class": submission.fault_class, "agent_reasoning": submission.reasoning, "contradictions": json.dumps(submission.contradictions())}
            else:
                fields = {"computed": None,"produced_value":None,"produced_value_units":None,"produced_metric_name":None,"produced_value_location":None,"produced_ranking_json":None,"produced_group_count":None,"operationalisation":None,"assumptions_json":None,"analysis_script":None,"could_not_compute_reason":None,"agent_fault_class":None,"agent_reasoning":None,"contradictions":"[]"}
            if limit_hit and not submission:
                fields.update(computed=0, could_not_compute_reason=f"budget exhausted ({limit_hit} limit) before submitting a result", agent_fault_class="agent_fault")
            source = "file" if evidence.get("location_kind") == "file" and evidence.get("evidence_text") else "none"
            text = evidence.get("evidence_text") if source == "file" else ""
            if source == "none" and fields["produced_value"] is not None:
                source = "transcript"; text = "\n".join(getattr(m,"text","") or "" for m in (sample.messages or []) if getattr(m,"role",None) == "tool")
            tokens_in,tokens_out,cost,_ = _usage_cost(experiment["model"], sample)
            await catalog.db.execute("UPDATE arm_b_runs SET computed=?,produced_value=?,produced_value_units=?,produced_metric_name=?,produced_value_location=?,produced_ranking_json=?,produced_group_count=?,operationalisation=?,assumptions_json=?,analysis_script=?,could_not_compute_reason=?,agent_fault_class=?,agent_reasoning=?,submission_json=?,submission_error=?,contradictions_json=?,evidence_source=?,evidence_text=?,setup_json=?,limit_hit=?,wall_clock_seconds=?,working_seconds=?,message_count=?,input_tokens=?,output_tokens=?,estimated_cost_usd=? WHERE custom_id=?", (fields["computed"],fields["produced_value"],fields["produced_value_units"],fields["produced_metric_name"],fields["produced_value_location"],fields["produced_ranking_json"],fields["produced_group_count"],fields["operationalisation"],fields["assumptions_json"],fields["analysis_script"],fields["could_not_compute_reason"],fields["agent_fault_class"],fields["agent_reasoning"],json.dumps(raw,sort_keys=True) if raw else None,error,fields["contradictions"],source,(text or "")[:20000],json.dumps(setup,sort_keys=True),limit_hit,sample.total_time,sample.working_time,len(sample.messages or []),tokens_in,tokens_out,cost,row["custom_id"]))
            parsed += 1
        await catalog.db.commit(); logger.info("arm_b_collected", experiment=args.experiment, runs=len(rows), parsed=parsed, without_log=missing)
    return 0


async def command_arm_b_verify(args: Any, logger: Any, Catalog: Any) -> int:
    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_b_schema(catalog); experiment = await get_experiment(catalog,args.experiment)
        rows = await (await catalog.db.execute("SELECT custom_id,paper_id,computed,produced_value,produced_value_location,evidence_source,evidence_text,operationalisation,analysis_script FROM arm_b_runs WHERE experiment_id=? AND status='succeeded'",(experiment["id"],))).fetchall()
        for row in rows:
            if not row["computed"]:
                note="not computed"; passed=None
            else:
                missing = [name for name in ("operationalisation","analysis_script") if not row[name]]
                present = row["produced_value"] is None or value_present(float(row["produced_value"]),row["evidence_text"])
                passed = int(bool(row["produced_value_location"]) and present and not missing)
                note = "value found at named location" if passed else "; ".join((["missing " + ", ".join(missing)] if missing else []) + ([] if row["produced_value"] is None or present else ["value not found at named location"]))
            await catalog.db.execute("UPDATE arm_b_runs SET location_verified=?,location_verify_note=? WHERE custom_id=?",(passed,note,row["custom_id"]))
        await catalog.db.commit(); logger.info("arm_b_verified",experiment=args.experiment,runs=len(rows))
    return 0


async def command_arm_b_compare(args: Any, logger: Any, Catalog: Any) -> int:
    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_b_schema(catalog); experiment=await get_experiment(catalog,args.experiment)
        rows=await (await catalog.db.execute("SELECT * FROM arm_b_runs WHERE experiment_id=? AND status='succeeded' ORDER BY paper_id,repetition_index",(experiment["id"],))).fetchall(); grouped: dict[int,list[Any]]={}
        for row in rows: grouped.setdefault(int(row["paper_id"]),[]).append(row)
        for paper_id,runs in grouped.items():
            withheld=await _withheld(catalog,experiment["claim_experiment_id"],paper_id); votes=["computed" if r["computed"] else "not_computed" for r in runs if r["computed"] is not None]; computed,vote_counts,tied=_modal(votes,("computed","not_computed")); ran=computed == "computed"
            # Put every repetition on one scale BEFORE aggregating. Paper 482's runs reported the
            # same 79.02% once as 0.790 (proportion) and once as 79.02 (percent); the median of
            # the raw pair was 39.9, a number no run produced. Normalise to proportion per run.
            def _as_proportion(r: Any) -> float:
                v = float(r["produced_value"])
                return v / 100.0 if (r["produced_value_units"] or "").strip().lower() in {"%", "percent", "pct", "percentage"} else v
            values=[_as_proportion(r) for r in runs if r["produced_value"] is not None]; disagreement=len({round(v, 6) for v in values})>1; produced=statistics.median(values) if values else None; spread=max(values)-min(values) if values else None
            ranking=next((json.loads(r["produced_ranking_json"]) for r in runs if r["produced_ranking_json"]),None); groups=next((r["produced_group_count"] for r in runs if r["produced_group_count"] is not None),None)
            units="proportion" if values else next((r["produced_value_units"] for r in runs if r["produced_value_units"]),None)
            comparison=compare_paper(withheld.get("claim_type"),produced,units,ranking,groups,withheld,ran,for_arm="B")
            faults=[r["agent_fault_class"] for r in runs if r["agent_fault_class"]]
            if comparison["fault"] == "value_mismatch":
                faults=["value_mismatch"]
            elif not faults and comparison["fault"]:
                faults=[comparison["fault"]]
            fault,fault_votes,fault_tied=_modal(faults,FAULT_PRECEDENCE)
            verified=[r["location_verified"] for r in runs if r["location_verified"] is not None]; location=int(all(verified)) if verified else None
            needs=int(tied or fault_tied or fault == "agent_fault" or location == 0 or disagreement or len(runs)<experiment["repeat_count"] or comparison["tolerance_verdict"] in {"not_comparable","underspecified"})
            wall=[r["wall_clock_seconds"] for r in runs if r["wall_clock_seconds"]]
            await catalog.db.execute("INSERT INTO arm_b_decisions(experiment_id,paper_id,claim_type,computed,computed_votes_json,fault_class,fault_votes_json,produced_value,produced_value_spread,produced_values_json,reported_value,reported_units,metric_name,scale_note,abs_delta,rel_delta,comparator,tolerance_verdict,qualitative_verdict,rrl_level,comparison_note,location_verified,value_disagreement,needs_adjudication,completed_run_count,wall_clock_seconds,aggregated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(experiment_id,paper_id) DO UPDATE SET computed=excluded.computed,computed_votes_json=excluded.computed_votes_json,fault_class=excluded.fault_class,fault_votes_json=excluded.fault_votes_json,produced_value=excluded.produced_value,produced_value_spread=excluded.produced_value_spread,produced_values_json=excluded.produced_values_json,reported_value=excluded.reported_value,reported_units=excluded.reported_units,metric_name=excluded.metric_name,scale_note=excluded.scale_note,abs_delta=excluded.abs_delta,rel_delta=excluded.rel_delta,comparator=excluded.comparator,tolerance_verdict=excluded.tolerance_verdict,qualitative_verdict=excluded.qualitative_verdict,rrl_level=excluded.rrl_level,comparison_note=excluded.comparison_note,location_verified=excluded.location_verified,value_disagreement=excluded.value_disagreement,needs_adjudication=excluded.needs_adjudication,completed_run_count=excluded.completed_run_count,wall_clock_seconds=excluded.wall_clock_seconds,aggregated_at=excluded.aggregated_at",(experiment["id"],paper_id,withheld.get("claim_type"),int(ran),json.dumps(vote_counts),fault,json.dumps(fault_votes),produced,spread,json.dumps(values),withheld.get("point_estimate"),withheld.get("units"),withheld.get("metric_name"),comparison["scale_note"],comparison["abs_delta"],comparison["rel_delta"],comparison["comparator"],comparison["tolerance_verdict"],comparison["qualitative_verdict"],_rrl(ran,comparison["tolerance_verdict"]),comparison["note"],location,int(disagreement),needs,len(runs),max(wall) if wall else None,utc_now()))
        await catalog.db.commit(); logger.info("arm_b_compare_complete",experiment=args.experiment,papers=len(grouped))
    return 0


async def command_arm_b_report(args: Any, logger: Any, Catalog: Any) -> int:
    from rich.console import Console
    from rich.table import Table
    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_b_schema(catalog); experiment=await get_experiment(catalog,args.experiment)
        decisions=await (await catalog.db.execute("SELECT d.*,p.collection_year y FROM arm_b_decisions d JOIN papers p ON p.id=d.paper_id WHERE d.experiment_id=? ORDER BY d.paper_id",(experiment["id"],))).fetchall()
        table=Table(title=f"Stage 4 Arm B — {args.experiment}"); [table.add_column(c) for c in ("paper","computed","produced","reported","verdict","conclusion","RRL","fault","location","flag")]
        for row in decisions: table.add_row(str(row["paper_id"]),"yes" if row["computed"] else "no","-" if row["produced_value"] is None else f"{row['produced_value']:g}","-" if row["reported_value"] is None else f"{row['reported_value']:g}",row["tolerance_verdict"],row["qualitative_verdict"],row["rrl_level"],row["fault_class"] or "-",{1:"ok",0:"FAIL"}.get(row["location_verified"],"-"),"ADJUDICATE" if row["needs_adjudication"] else "")
        Console().print(table)
        payload={"experiment":args.experiment,"papers":len(decisions),"scope_papers":list(ARM_B_PAPERS),"skips":ARM_B_SKIPS,"verdicts":dict(Counter(r["tolerance_verdict"] for r in decisions)),"no_rate_reported":"stage4-spec.md §8 forbids a reproduction rate at this n"}
        if getattr(args,"output",None): Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(payload,indent=2,sort_keys=True),encoding="utf-8")
        logger.info("arm_b_report",**payload)
    return 0


async def command_arm_b_contrast(args: Any, logger: Any, Catalog: Any) -> int:
    from rich.console import Console
    from rich.table import Table
    async with Catalog(Path(args.database)) as catalog:
        await ensure_arm_b_schema(catalog)
        arm_a=await (await catalog.db.execute("SELECT id FROM arm_a_experiments WHERE name=?",(args.arm_a,))).fetchone(); arm_b=await get_experiment(catalog,args.arm_b)
        if arm_a is None: raise ValueError(f"Unknown Arm A experiment: {args.arm_a}")
        arows=await (await catalog.db.execute("SELECT * FROM arm_a_decisions WHERE experiment_id=?",(arm_a["id"],))).fetchall(); brows=await (await catalog.db.execute("SELECT * FROM arm_b_decisions WHERE experiment_id=?",(arm_b["id"],))).fetchall(); ad={int(r["paper_id"]):r for r in arows}; bd={int(r["paper_id"]):r for r in brows}
        table=Table(title=f"Stage 4 contrast — A: {args.arm_a}; B: {args.arm_b}"); [table.add_column(c) for c in ("paper","A rung","A verdict","A RRL","B computed","B verdict","B RRL","note")]
        for paper_id in sorted(set(ad)|set(bd)|set(ARM_B_SKIPS)):
            a,b=ad.get(paper_id),bd.get(paper_id); note=ARM_B_SKIPS.get(paper_id,"")
            table.add_row(str(paper_id),a["intervention_rung"] if a else "-",a["tolerance_verdict"] if a else "-",a["rrl_level"] if a else "-","yes" if b and b["computed"] else ("no" if b else "skip"),b["tolerance_verdict"] if b else "-",b["rrl_level"] if b else "-",note)
        Console().print(table); logger.info("arm_b_contrast",arm_a=args.arm_a,arm_b=args.arm_b,rows=len(set(ad)|set(bd)|set(ARM_B_SKIPS)))
    return 0
