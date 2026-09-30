# Third-party material

## Agentic reproducibility harness (Stage 4 sandbox)

`arm_a/Dockerfile` is copied verbatim, and the image-viewer tool and tool set in `arm_a/task.py`
are adapted, from <https://github.com/tobihol/agentic-reproducibility> (commit `dde239c`,
`pipeline/agentic_analysis/`).

```
MIT License

Copyright (c) 2026 The authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ARA (Stage 2)

Stage 2 calls the ARA pipeline from
<https://github.com/AndresLaverdeMarin/agentic_reproducibility_assessment> at commit
`e015be37d82c57e6da826ad50fe8023e9f1255f5` (GPL-3.0). It is **not** included here. You clone
it into `.external/ara` yourself (see the README), and `ara_bridge.py` runs it in a separate
process.

## Human coding benchmark

`baseline/haim-lak/artifacts/qualtrics-survey-data.csv` and `explanations.pdf` are the released
coding data of Haim et al. (2023), *How to Open Science: A Principle and Reproducibility
Review of the Learning Analytics and Knowledge Conference*, redistributed unchanged under
CC BY 4.0 (`baseline/haim-lak/artifacts/CONTENT-LICENSE`).
