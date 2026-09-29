# Butcher Chess Engine

An educational policy-only neural chess engine. Predicts the next move
directly from a position using a deep convolutional network. No Monte Carlo
Tree Search, no value head, no tablebase. Inspired by the *architecture
style* of AlphaZero-style policy networks — not by LC0, which uses MCTS,
a value head, and a full self-play reinforcement pipeline.

## Key Features

### Neural Network Architecture
- Deep convolutional network with residual blocks
- Input: `8×8×18` tensor representation of a chess position
- Output: **4672 raw logits** (AlphaZero-style move encoding, no softmax
  inside the model)
- Illegal moves are masked with `-1e9` before softmax — applied outside
  the model, both at training and inference
- L2 weight decay (`1e-4`) and dropout (`0.1` / `0.2`) for regularization
- Batch normalization in every block

### Training Capabilities
- Supervised training on chess puzzles (PGN)
- Deterministic train/eval split by `md5(FEN)` — 90% / 10%
- Configurable via CLI: `--batch`, `--lr`, `--epochs`, `--max-puzzles`
- Manual checkpointing: model is saved to disk **every N epochs**, not
  automatically on interruption
- Resume from a saved `.keras` file

### What Is NOT Implemented
- ❌ MCTS or any tree search
- ❌ Value head (position evaluation)
- ❌ Working self-play reinforcement — the code exists in
  `self_play/game_simulator.py` and `training/self_play_trainer.py`, but
  without a value head and without exploration it does not improve the
  model
- ❌ Endgame tablebase
- ❌ Automatic checkpointing on crash

## Puzzle Accuracy

Held-out set, deterministic split by `md5(FEN)`, **n = 3044**.
Model: 30 epochs, ~27k training puzzles, batch 64, lr 1e-3.

| Metric | Value | Notes |
|--------|-------|-------|
| Top-1 | **20.93%** | correct move is argmax |
| Top-3 | **39.39%** | correct move in top-3 |
| Top-5 | **48.88%** | correct move in top-5 |
| Top-10 | **64.29%** | correct move in top-10 |
| NLL | **3.10** | mean −log p(correct) |

Reference points:

| Baseline | Top-1 | NLL |
|----------|-------|-----|
| Uniform over ~27 legal moves | ≈ 3.7% | ≈ log(27) ≈ 3.30 |
| Butcher 0.2.0 | **20.93%** | **3.10** |

The model is roughly 5× better than random at picking the exact puzzle
solution, and its NLL is below the uniform baseline — meaning it assigns
non-trivial probability mass to the correct move.

## Accuracy by Puzzle Rating

| Rating | n | Top-1 | Top-3 |
|--------|---|-------|-------|
| 800–999 | 754 | 24.4% | 42.0% |
| 1000–1199 | 825 | 22.7% | 40.8% |
| 1200–1399 | 564 | 19.3% | 38.5% |
| 1400–1599 | 526 | 19.0% | 37.6% |
| 1600–1799 | 374 | 15.2% | 34.8% |
| 1800–1999 | 1 | 0.0% | 0.0% |

Top-1 declines monotonically with puzzle difficulty (24.4% → 15.2%). This
is consistent with a model that has learned real patterns rather than
memorized a specific rating band. It is *not* proof of zero overfitting —
only an indication.

## ELO

**Butcher 0.2.0 has no official ELO.** No matches were played against a
fixed opponent pool, so any single number would be fabricated.

| Question | Answer |
|----------|--------|
| Has it played rated games? | No |
| Against Stockfish? | Not yet |
| Against classical engines? | No |
| On Lichess / chess.com? | No |
| Puzzle accuracy → playing ELO? | Not a valid conversion |

**Why no estimate is given:**

- **Puzzle rating ≠ playing strength.** Lichess puzzle rating measures
  how hard a tactic is to spot in a given position, not how well a player
  performs across a full game. Solving 21% of 800–1800 puzzles says
  nothing directly about opening, endgame, or defensive skill.
- **The model has never trained on full games.** The training data are
  puzzles that start from mid-game positions. Playing from the starting
  position is out of distribution.
- **No search, no value.** With one-ply policy and argmax, the engine
  cannot see forced continuations. Its practical strength is bounded by
  what the raw policy can rank.

To produce a real ELO, the project would need:

1. A fixed opponent pool (e.g. Stockfish skill levels 0–10, or prior
   Butcher versions).
2. Hundreds of games per pairing, colors alternating.
3. BayesElo or Ordo rating calculation.

None of this has been done for 0.2.0.

**Caveats:**

- The model is trained on puzzles, not full games. It has never seen a
  starting position during training.
- No value head and no search: the model cannot see forced continuations
  beyond one ply.
- Puzzle-solving ELO and playing-strength ELO are different scales. Top-1
  of 21% on Lichess puzzles does not translate directly to playing strength.
