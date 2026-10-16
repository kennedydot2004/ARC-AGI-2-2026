# ARC-AGI-2 2026

An open, reproducible solver for the ARC Prize 2026 ARC-AGI-2 Kaggle
competition.

The project prioritizes task-level generalization: infer a transformation from
the demonstrations in each task, verify it against every demonstration, and
apply only consistent hypotheses to the unseen test input(s). Predictions must
be exact grids and the Kaggle adapter must emit two attempts for every test
output.

## Competition constraints

- Submit a Kaggle notebook that writes `submission.json`.
- Include all task IDs and preserve the order of multiple test inputs.
- Provide `attempt_1` and `attempt_2` for each test output.
- Run with internet disabled and finish within the competition's 12-hour limit.
- Publicly available external data and pretrained models are permitted.
- For prize eligibility, author-created code and methods must be open-sourced
  under a permissive public-domain license such as CC0 or MIT-0.

## Development status

The repository is being bootstrapped. The first milestone is a deterministic,
testable symbolic baseline and a validation harness over public ARC-AGI-2
tasks. No claims about leaderboard performance are made until measured on
held-out tasks.

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
finds `arc-agi_test_challenges.json`, runs the solver offline, and writes
`/kaggle/working/submission.json`. Rebuild its embedded source after solver
changes:

```powershell
python kaggle/build_submission_notebook.py
```
