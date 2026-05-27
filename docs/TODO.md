# TODO

Tracking gaps that matter for benchmarking and reproducibility.

1. **Hellaswag harness-style scoring** — Add an evaluation path aligned with [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) Hellaswag: rank the four `endings` by conditional log-likelihood (LM head), report `acc` / `acc_norm` (and document split + preprocessing parity). Keep the existing generative letter task as an optional variant so numbers are comparable to papers and harness leaderboards.
