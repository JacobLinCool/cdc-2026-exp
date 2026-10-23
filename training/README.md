# Lightweight learned evaluator

A 64 → 16 → 1 ReLU network learns a correction to the existing static evaluation. It has 1,057 float32 parameters; the versioned model file is 4,236 bytes. Rust inference needs no additional dependencies. The existing search, rules, move ordering, and tablebases remain in use. This is an evaluator with search, not a standalone move policy.

The default engine still uses its original evaluator. Load a trained network explicitly through the arena's `model=` setting or `Searcher::set_evaluator`.

An additional [exact-initialized structured evaluator experiment](structured/README.md) reconstructs the original numeric evaluation, expands its parameters, and trains it with explicit surrogate gradients. Its first 1,360-byte candidate did not improve held-out play: 47.56% against the original engine and 44.31% against this residual network, 800 games each. The prototype, exact initialization/export checks and negative results are retained for further experiments.

The [stronger-teacher follow-up](ranking/README.md) produced a 1,360-byte candidate that beats the original engine at 25 ms/move: **53.50%** score (95% interval 50.94%–56.06%, 800 games). It scored **50.81%** against this residual model (48.25%–53.38%, 800 games), which establishes neither superiority nor equivalence. Direct inference takes about 39% less time per evaluation than the residual model in the local microbenchmark. Load it with `structured=training/ranking/value.cse`; the scalar-fitting candidate won selection over preference fitting.

The [scaling sweep](scaling/README.md) and [data/capacity interaction supplement](scaling-joint/README.md) complete **9 conditions, 27 fits and 6,040 games**. More data and larger composition tables improve teacher-label fitting, but no new model establishes superiority over the preceding structured evaluator. The frozen candidate scores 49.88% against it at 25 ms (47.38%–52.31%, 800 games). All results, including the negative comparisons, are retained. The larger models use exact-initialized fixed-point lookup corrections (`expanded=MODEL.cxe`); the arena also supports explicit node budgets. See the [combined results](scaling-overview.json) and [data/capacity figure](scaling-grid.png).

The [capacity × test-time study](test-time/README.md) adds **2,592 games** at 10, 40, and 160 ms per move, plus equal-node controls. Against a fixed 40 ms opponent, the 139-, 499-, and 1,948-parameter models gain approximately 140, 145, and 109 Elo from 10 to 160 ms. Additional search helps, but a larger capacity-specific gain remains unproven: the primary direct-match score change is +6.77 percentage points (95% interval −1.04 to +14.58). All nine existing models are frozen, with matched training data and three training seeds. This controlled study fixes the flip-search threshold and disables tablebases; it establishes no asymptotic ceiling or production-setting winner. See the [curves](test-time/test-time-scaling.png) and [results](test-time/results.json).

The [499-parameter refinement](improve499/README.md) keeps the selected 499 architecture and tests six training recipes, with **18 fits and 2,752 games**. The selected deeper-value-plus-ranking model scores **49.75%** against the frozen 499 baseline at 40 ms (95% interval 47.67%–51.75%, 1,200 fresh games), and **48.50%** at 160 ms (44.88%–52.12%, 400 fresh games). Neither comparison establishes an improvement. Improved teacher fitting did not produce a confirmed strength gain, so the existing 499 baseline is retained for subsequent work. See the [full results](improve499/results.json) and [figure](improve499/results.png).

The [four-budget tournament](tournament1600/README.md) completes **1,792 games**, including a 480-game extension at 1,600 ms. Five frozen versions compete at **25, 100, 400, and 1,600 ms** per move and against the original fixed at 25 ms. Round-robin point leaders are 499 (53.9%), 1,948 (57.0%), 139 (53.5%), and 499 (54.7%), respectively, with 128 games per model per budget. At 1,600 ms, 499 versus 139 scores **67.2%** (15 wins, 13 draws, 4 losses; pointwise 95% interval 56.2%–78.1%), compared with 50% at 400 ms. Its score against the original at equal 1,600 ms is 56.2% (42.2%–70.3%), so no universal winner is established.

Against the original fixed at 25 ms, 499 scores 75.0% at both 400 and 1,600 ms; its paired gain interval is −15.6 to +15.6 percentage points. This does not establish a ceiling. All between-model gain-difference intervals include zero for both 400-to-1,600 and 25-to-1,600 ms, so an additional gain from larger capacity remains unconfirmed. The 499 research baseline is retained. These are exploratory comparisons with 16 paired deals per cell and no multiplicity correction; training histories differ, and the production flip-depth and successor-tablebase thresholds also change at the longer budget. See the [comparison](tournament1600/comparison.png), [head-to-head matrices](tournament1600/head-to-head.png), [full statistics](tournament1600/results.json), and [independent reconciliation](tournament1600/verification.json). The preceding three-budget artifacts remain unchanged.

## Held-out result

The selected model, [`strong-teacher.nn`](strong-teacher.nn) with `model_scale=0.5`, beat the default evaluator on the predeclared holdout at **25 ms per move**:

| Metric | Result |
| --- | ---: |
| Games / independent paired deals | 800 / 400 |
| Wins / draws / losses | 312 / 236 / 252 |
| Score (draw = half a point) | **53.75%** |
| Deal-paired bootstrap 95% score interval | **51.13%–56.38%** |
| Relative Elo estimate | +26.1 |
| Actual thinking time, model / baseline | 21.886 / 21.975 ms per move |
| Model parameters / file size | 1,057 / 4,236 bytes |

The interval is entirely above 50%, satisfying the frozen success criterion. This establishes an advantage at this tested budget against this baseline; longer time controls were not tested. The model is a learned evaluator used by the existing search, and the reported size excludes the search engine and tablebases.

The selected checkpoint used 57,694 training positions and 13,691 validation positions from separate games in the 40 ms self-play corpus. The final 400 deal seeds appeared in neither training, checkpoint validation, nor development matches. The candidate and 800-game sample size were frozen before evaluation, with no optional stopping.

Evidence: [final report](holdout.json), [all 800 game records](holdout.csv), [frozen candidate](frozen-candidate.json), and [training metadata](strong-teacher.json).

To use the evaluator in a fresh arena match:

```sh
SEED=5000000 ARENA_CSV=training/new-match.csv \
  engine/target/release/arena 100 25 \
  'model=training/strong-teacher.nn,model_scale=0.5' '' 6
```

To reproduce training of this candidate from the locally retained dataset:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 training/train.py training/strong.csv training/retrained.nn \
  --epochs 100 --outcome-weight 0
```

At the time of this first residual-network experiment, all 13 Rust tests and the `wasm32-unknown-unknown` build check passed. The later structured follow-up passes 19 tests.

## Experiment record

This run generated 98,134 public-position records from 1,408 self-play games: 384 games at 12 ms/move and 1,024 at 40 ms/move. The two corpora were trained separately. [`run.json`](run.json) records seeds, hashes, tools, and the starting commit. [`experiment-plan.json`](experiment-plan.json) fixes the final test protocol. Per-candidate JSON and arena CSV files preserve all development results.

## Training

Run from the repository root with Rust and Python 3 + `training/requirements.txt` installed:

```sh
cargo build --release --manifest-path engine/Cargo.toml --bin neural_data --bin arena
engine/target/release/neural_data training/selfplay.csv 384 12 6 2000000
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python3 training/train.py training/selfplay.csv training/value.nn --epochs 100
```

`neural_data OUTPUT GAMES MS THREADS FIRST_SEED` plays the baseline against itself. Each CSV row is `game_seed,baseline_score,search_score,outcome,64_features`. It samples every second ply when iterative deepening completed a depth. Truncated games are discarded. The network sees only `Pos`: visible pieces, publicly inferable piece counts, threats, safe mobility, static score, and phase/progress counters. The secret deal is never passed to feature extraction or inference.

The trainer holds out 20% of whole games, chooses the checkpoint with the lowest validation cross-entropy, and exports weights, a metadata JSON, and a NumPy/Rust parity fixture. Its target is a configurable mixture of the search score's logistic probability (`score / 600`) and the final game outcome. `--outcome-weight 0` uses search labels only. The residual is regularized toward zero. No accelerator is required. `--quiet-only` filters to positions with no immediate captures for either side and zeros the unused threat-channel weights so they cannot introduce untrained behavior during inference.

The larger dataset can be regenerated with `engine/target/release/neural_data training/strong.csv 1024 40 6 2100000`.

Training CSVs remain local and are ignored by Git. Metadata records their SHA-256, the exact training command, NumPy version, game split, and checkpoint loss. Given the same CSV and environment, training uses a fixed RNG seed. Regenerating data is not bit-reproducible because the teacher uses wall-clock search budgets and parallel workers finish in different orders.

## Evaluation

Use paired deals with both seating orders, separate seeds from the training/validation games, and equal time limits. For example:

```sh
SEED=3000000 ARENA_CSV=training/dev-teacher.csv \
  engine/target/release/arena 120 25 \
  'model=training/teacher.nn,model_scale=0.5' '' 6
python3 training/report.py training/dev-teacher.csv \
  --model training/teacher.nn --scale 0.5 --purpose development --time-ms 25
```

`model_scale` multiplies the learned correction. Both players use the same engine search and tablebases. Inference runs inside the player's search time budget. CSVs include actual total thinking time and move counts, including tablebase construction; the engine's time limits are soft.

Use development deals to choose the target and correction strength. Freeze the candidate and sample size before a final test on fresh deals. The report bootstraps complete deal pairs (20,000 resamples), checks training/evaluation seed separation, and verifies the model hash against its training metadata. Score is `(wins + draws / 2) / games`. A positive result requires the held-out 95% interval to lie entirely above 50%; a better training loss or a small development lead is insufficient.

## Format and checks

`CDCNN001` is eight magic/version bytes followed by 1,057 little-endian float32 values: 16×64 input weights in hidden-unit-major order, 16 hidden biases, 16 output weights, and one output bias. Prediction is `baseline + scale * 600 * network(features)`, rounded and clamped to the search's static evaluation bounds. Corrupt, non-finite, unsupported-version, or wrong-sized models fail explicitly.

```sh
cargo test --manifest-path engine/Cargo.toml
cargo check --manifest-path engine/Cargo.toml --target wasm32-unknown-unknown --features wasm --lib
```

Tests cover hidden-layout isolation, color/reflection invariance, corrupt model rejection, zero-residual behavior, and agreement with NumPy inference. Arena games assert move legality and use actual rule termination rather than an artificial draw at a fixed move cap.

## Development results

All rows below are development comparisons, not independent proof of superiority. Each 120-game set uses the same 60 deals with exchanged seats; the 400-game confirmation uses a different 200 deals. All use 25 ms/move. Repeated selection on these sets makes a fresh holdout necessary.

| Candidate | Scale | Games | W / D / L | Score | Paired 95% interval |
| --- | ---: | ---: | --- | ---: | --- |
| [dev-full](dev-full.json) | 1.0 | 120 | 37 / 29 / 54 | 42.92% | 36.25–49.58% |
| [dev-quarter](dev-quarter.json) | 0.25 | 120 | 44 / 34 / 42 | 50.83% | 44.17–57.50% |
| [dev-teacher](dev-teacher.json) | 0.5 | 120 | 48 / 36 / 36 | 55.00% | 48.75–60.83% |
| [dev-teacher-full](dev-teacher-full.json) | 1.0 | 120 | 46 / 32 / 42 | 51.67% | 45.83–57.50% |
| [dev-teacher-quarter](dev-teacher-quarter.json) | 0.25 | 120 | 38 / 32 / 50 | 45.00% | 38.75–51.67% |
| [dev-outcome](dev-outcome.json) | 0.25 | 120 | 39 / 35 / 46 | 47.08% | 41.67–52.50% |
| [dev-quiet](dev-quiet.json) | 0.5 | 120 | 42 / 35 / 43 | 49.58% | 43.33–56.25% |
| [dev-teacher-confirm](dev-teacher-confirm.json) | 0.5 | 400 | 155 / 97 / 148 | 50.88% | 47.62–54.25% |
| [dev-strong-teacher](dev-strong-teacher.json) | 0.5 | 120 | 53 / 32 / 35 | 57.50% | 49.58–65.42% |
| [dev-strong-mixed](dev-strong-mixed.json) | 0.5 | 120 | 48 / 38 / 34 | 55.83% | 49.58–62.08% |
| [dev-strong-quiet](dev-strong-quiet.json) | 0.5 | 120 | 38 / 37 / 45 | 47.08% | 41.67–52.92% |
