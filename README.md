# ARC-AGI-2 2026

An open, reproducible solver for the ARC Prize 2026 ARC-AGI-2 Kaggle
competition.

The project uses an offline hybrid approach: a local language model proposes
short programs from a whitelisted transformation catalog; the solver executes
each program and accepts it only if it exactly reproduces every demonstration.
Predictions must be exact grids and the Kaggle adapter emits two attempts for
every test output.

## Competition constraints

- Submit a Kaggle notebook that writes `submission.json`.
- Include all task IDs and preserve the order of multiple test inputs.
- Provide `attempt_1` and `attempt_2` for each test output.
- Run with internet disabled and finish within the competition's 12-hour limit.
- Publicly available external data and pretrained models are permitted.
- For prize eligibility, author-created code and methods must be open-sourced
  under a permissive public-domain license such as CC0 or MIT-0.

## Development status

The initial symbolic baseline is verified but not competitive: on the full
120-task public evaluation set it solved 0 of 167 test outputs. The hybrid
model-proposal path is implemented and guarded by exact demonstration checks,
but model accuracy, latency, and GPU memory use have not yet been measured;
those require running the Kaggle notebook with attached weights.

## Local development

Run regression tests using only the Python standard library:

```powershell
python -m unittest discover -s tests -v
```

Evaluate against a labeled ARC task directory (for example, a local copy of
`data/evaluation` from the public `arcprize/ARC-AGI-2` repository):

```powershell
python -m arc_solver.evaluate data/ARC-AGI-2/data/evaluation
```

The public benchmark data is not committed here. Keep validation artifacts and
downloaded datasets under the ignored `data/` directory. Avoid repeatedly
tuning against public evaluation labels; prefer task-level splits from the
training set for iterative development.

See [docs/approach.md](docs/approach.md) for the current method, measured
limitations, validation protocol, and research direction.

## Kaggle notebook

`kaggle/arc_prize_2026_submission.ipynb` is a self-contained notebook that
finds `arc-agi_test_challenges.json`, loads an attached local Qwen3 model,
proposes and verifies programs offline, and writes
`/kaggle/working/submission.json`. Attach the
[Qwen3-30B-A3B-Instruct-2507 Kaggle Model](https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/30b-a3b-instruct-2507)
as a notebook input. The model is Apache-2.0 licensed and requires
Transformers 4.51+; the 30.5B-parameter BF16 weights require multi-GPU memory,
so the notebook is intended for the competition's L4x4 accelerator. Actual
Kaggle runtime compatibility, memory use, and inference performance still need
to be measured. Rebuild its embedded source after solver changes:

```powershell
python kaggle/build_submission_notebook.py
```
