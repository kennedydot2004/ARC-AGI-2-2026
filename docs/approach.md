# Solver strategy and validation

## Current design

Each task is solved independently. We enumerate a small library of grid
transformations and search compositions to depth two by default. An optional
bounded beam search to depth three is experimental: on the initial 200-task
training holdout it found no additional exact outputs and took about 542
seconds, compared with about 52 seconds for the depth-two baseline. The beam
keeps 128 joint training-state candidates and at most three short program
paths per state. An optional local Qwen model proposes up to three additional
sequences, up to five operations long.
The operation catalog covers geometric transforms, global color maps,
translations inferred from demonstrations, cropping, connected-component
selection, simple color masking, and tiling/scaling. Every searched or proposed
program is executed on **all** demonstrations; only exact matches survive.
The model proposes several programs in one generation call instead of making
three separate calls. For a direct grid prediction, it sees all but the final
training pair, predicts that withheld output plus the actual test outputs, and
the solver uses its test guesses only if the withheld prediction exactly
matches the known training output. Program proposals remain checked against
every training pair, including the withheld one. This leave-one-example-out
check is a confidence filter, not proof that a test prediction is correct.
An accepted model program is ranked ahead of purely symbolic alternatives,
while the second attempt retains a distinct alternative when available. If no
program is consistent, the solver emits the unchanged input as a valid
fallback and records proposal rejection diagnostics.

This is a correctness-first scaffold, not a strong ARC solver. Passing examples
is necessary but not sufficient: multiple rules can fit a few demonstrations,
and this small library does not model many relational, compositional, or
context-dependent tasks. On the full 120-task public evaluation set, this
baseline solved 0 of 167 test outputs and found no demonstration-consistent
programs. It took 365 seconds on the development machine. This is a measured
negative baseline, not a leaderboard estimate; no task-specific rules were
added from the public-evaluation labels. A deterministic 200-task holdout
from the public training set scored 5/212 exact outputs (2.36%) with the
depth-two symbolic solver. Experimental depth-three beam search found no
additional hits on that same sample and was about ten times slower, so it
remains optional rather than the Kaggle default. The calibrated direct
prediction path scored 0.00 on Kaggle: 1 of 240 tasks passed calibration and
5 of 720 candidate programs verified. `operations_extra.py` then added symmetry,
gravity, enclosed fill, grid-split boolean rules and learned local rules. On two
disjoint 200-task training samples (every fifth task from offsets 0 and 2) they
solved 21 tasks versus 7 before, at roughly four times the runtime.

## Why the next step is not just a bigger brute-force search

The official ARC Prize 2025 results report a 24% private ARC-AGI-2 score for
the top Kaggle entry and a 54% score for a refinement system evaluated at
substantially higher per-task cost. Those results motivate a verified
refinement loop: propose a rule, execute it on every demonstration, reject
inconsistent rules, and use the counterexamples to improve the next proposal.
They do not imply that a small fixed transformation catalog will score well.

Kaggle's competition-specific External Data and Tools rule allows external data
and models unless prohibited by the host, subject to public/equitable access or
the rule's reasonableness standard. Final inference is offline, so model
weights must be attached locally to the notebook; API calls and runtime
downloads are not viable. The initial model choice is [Qwen3-30B-A3B-Instruct-2507 on Kaggle](https://www.kaggle.com/models/qwen-lm/qwen-3/Transformers/30b-a3b-instruct-2507),
licensed under Apache-2.0 and requiring Transformers 4.51+. Its 30.5B total
parameters imply roughly 61 GB for BF16 weights before runtime overhead; the
competition's L4x4 accelerator advertises 96 GB across four GPUs. This is a
feasibility estimate, not a measured fit or speed guarantee. The public Kaggle
model listing is available as an attachable notebook input, but compatibility
with the competition image still needs runtime verification. The symbolic
verifier remains the authority: model proposals must use only whitelisted
operations and match all demonstrations before they can become candidates.

## Validation protocol

1. Run the deterministic unit tests for grid validity, task ordering, candidate
   inference, and the exact Kaggle output contract.
2. Iterate on task-level validation splits from the 1,000 public training
   tasks. Keep task IDs disjoint between development and measurement splits.
3. Run the public evaluation set as a held-out benchmark, record both exact
   output accuracy and all-outputs-per-task accuracy, and do not add
   task-specific rules based on its labels.
4. Rebuild and inspect the self-contained notebook, then run it with Kaggle's
   internet-disabled setting and verify the actual output file, all task IDs,
   multiple test inputs, proposal acceptance/rejection counts, runtime, and
   memory use.

The score that matters is exact-grid accuracy on hidden test outputs. Matching
some cells, returning a plausible shape, or reproducing a training example is
not partial credit.

## External references

- [ARC Prize 2026 ARC-AGI-2 competition](https://arcprize.org/competitions/2026/arc-agi-2)
- [ARC-AGI-2 dataset and task format](https://github.com/arcprize/ARC-AGI-2)
- [ARC Prize 2025 results and analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [ARC Lang refinement-loop example](https://github.com/jerber/arc-lang-public)
- [Kaggle competition rules, External Data and Tools](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)
