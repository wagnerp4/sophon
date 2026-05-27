# models

HF checkpoints stay here as directories named like Hub ids with slashes replaced by dashes (for example `google-gemma-4-31b-it`). Each checkpoint directory must contain `config.json` at its root plus weight shards.

Presets in `mithril.backend.hf.registry` resolve default paths such as `models/google-gemma-4-31b-it` relative to the mithril repo root (`mithril/pyproject.toml` parent).

Inference and `mithril-benchmark` resolve weights in this order:

1. `--model PATH` when passed on the CLI.
2. `GEMMA4_MODEL` (single checkpoint directory).
3. `GEMMA4_LOCAL_DIR` (defaults to `./models/meta-llama-Llama-2-7b-chat-hf` when unset in code paths that use `DEFAULT_LOCAL_DIR`).

Weight blobs under `models/` are gitignored; only this README is tracked.
