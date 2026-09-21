# Adapters

Layout:

```text
data/training/adapters/<preset_key>/<dataset_id>/<YYYYMMDD_HHMMSS>/
  adapter_config.json
  adapter_model.safetensors
  tokenizer files
  run_meta.json
  recipe.resolved.yaml
  metrics.jsonl
  trainer_output/
  smoke_eval.json
data/training/adapters/index.json
```

`data/*` is gitignored except README, assets, and benchmarks. Adapters are local artifacts.

## Names

`run_id` is a timestamp. The index name is `output_name` when set, otherwise `{dataset}-{run_id}`. Stages use `{output_name}-{stage_id}`. Name collisions append `-2`, `-3`. Run directories are never overwritten.

`index.json` records `name`, `preset`, `dataset`, `run_id`, `path`, `created`, `last_loss`, `steps`, `parent`, `stage_id`.

## Select

- `/adapter list` reads the index
- `/adapter load NAME|PATH|run_id` then reloads the matching HF preset with `PeftModel.from_pretrained`
- `/adapter show` / `/adapter clear`
- Ctrl+M: collapsible **adapters** group (name · preset · dataset · date)
- `/eval train` lists the same index
- `/finetune-status [NAME]` reads `run_meta.json` and plots `metrics.jsonl` as a sparkline

After `/finetune` the session unloads weights and **does not** attach the adapter. The job prints `/adapter load NAME`. Isolation matches [VISION.md](../VISION.md): adapters stay in their own directory. The base preset on disk is not overwritten.
