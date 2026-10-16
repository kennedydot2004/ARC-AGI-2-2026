# Solver strategy and validation

## Current design

Each task is solved independently. We enumerate a small library of grid
transformations, keep only programs that reproduce every demonstration, and
rank their predictions for the task's unseen inputs. The current library covers
geometric transforms, global color maps, translations inferred from the
demonstrations, cropping, connected-component selection, simple color masking,
and tiling/scaling. Programs are searched to depth two. If no program is
consistent, the solver emits the unchanged input as a valid fallback rather
than inventing an unsupported output.

This is a correctness-first scaffold, not a strong ARC solver. Passing examples
is necessary but not sufficient: multiple rules can fit a few demonstrations,
and this small library does not model many relational, compositional, or
context-dependent tasks. On the full 120-task public evaluation set, this
baseline solved 0 of 167 test outputs and found no demonstration-consistent
programs. It took 365 seconds on the development machine. This is a measured
negative baseline, not a leaderboard estimate; no task-specific rules were
added from the public-evaluation labels.

## Why the next step is not just a bigger brute-force search

The official ARC Prize 2025 results report a 24% private ARC-AGI-2 score for
the top Kaggle entry and a 54% score for a refinement system evaluated at
substantially higher per-task cost. Those results motivate a verified
refinement loop: propose a rule, execute it on every demonstration, reject
inconsistent rules, and use the counterexamples to improve the next proposal.
They do not imply that a small fixed transformation catalog will score well.

For this offline competition, any learned or language-model component must use
publicly available, redistributable weights available to the notebook at
runtime. API calls and runtime downloads are not viable. The symbolic verifier
should remain the authority: model proposals must produce valid grids and
match all demonstrations before they can become prediction candidates.

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
   multiple test inputs, runtime, and memory use.

The score that matters is exact-grid accuracy on hidden test outputs. Matching
some cells, returning a plausible shape, or reproducing a training example is
not partial credit.

## External references

- [ARC Prize 2026 ARC-AGI-2 competition](https://arcprize.org/competitions/2026/arc-agi-2)
- [ARC-AGI-2 dataset and task format](https://github.com/arcprize/ARC-AGI-2)
- [ARC Prize 2025 results and analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [ARC Lang refinement-loop example](https://github.com/jerber/arc-lang-public)
