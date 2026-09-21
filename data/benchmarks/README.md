# benchmarks

Task manifests for `sophon-benchmark`. Each YAML file describes one benchmark: where the rows live, how to render a prompt, and how to score the model output.

`sophon-benchmark --task <id>` resolves `<id>` to `<this-folder>/<id>.yaml` and runs the task with the loaded local model.

## Available manifests

| File | Source | Notes |
| --- | --- | --- |
| `hellaswag.yaml` | `data/hellaswag/data/*.jsonl` (in repo) | Split-aware loader under `sophon.eval.datasets.hellaswag`. Default split `val`. Public `test` JSONL has no labels and is rejected for scored runs. |
| `mmlu.yaml` | `data/test/data/{dev,test}/*.csv` (NOT in repo) | Hendrycks MMLU. Download `data.tar` from the [MMLU repo README](../test/README.md) and extract into `data/test/data/` first. |
| `rag_ablation.yaml` | `data/benchmarks/rag/ablation.jsonl` | No-RAG vs default LEANN corpus. Also `/eval rag` in chat. |

## Manifest schema

```yaml
id: string                 # task id used on the CLI (matches the file stem)
description: string

source:
  kind: jsonl | mmlu_csv | hellaswag
  options: {}               # optional; dataset-specific (see Hellaswag below)
  # jsonl:
  path: relative/path/to/file.jsonl
  # mmlu_csv:
  root: relative/path/to/data
  splits: { dev: dev, test: test }
  # hellaswag:
  root: relative/path/to/data
  splits: { val: hellaswag_val.jsonl, train: hellaswag_train.jsonl }

prompt:
  type: multiple_choice | freeform
  system: optional string          # routed as the "system" chat message
  template: |                      # python str.format() template
    Question: {question}
    A. {a}
    ...
  field_map:                       # how raw row keys map to template placeholders
    question: q                    # template "{question}" reads row["q"]
    choices: endings               # for MC tasks; list[str] of options
    label: label                   # int index OR letter "A".."D"

scorer:
  name: multiple_choice_letter | gsm8k_final | exact_normalize
  num_choices: 4                   # MC tasks only

defaults:
  max_new_tokens: 64
  temperature: null
  top_p: null
  top_k: null
  max_examples: 200                # null = full split
  fewshot: 0                       # currently informational; v1 runs zero-shot
```

Paths under `source` resolve relative to the manifest file unless absolute.

### Hellaswag `source.options`

| Key | Type | Meaning |
| --- | --- | --- |
| `ctx_mode` | `raw` / `lm_eval` | `raw` uses dataset `ctx`. `lm_eval` rebuilds context from `ctx_a` + capitalized `ctx_b` like EleutherAI lm-eval preprocessing. |
| `lm_eval_normalize` | bool | Applies WikiHow bracket stripping and spacing cleanup (`lm_eval.tasks.hellaswag.utils.preprocess` behaviour) to activity, context, and endings when true. |
| `split_types` | list[str] or null | If set, keep only rows whose `split_type` is in this list (for example `indomain`, `zeroshot`). |
| `shuffle_choices_seed` | int or null | If set, permute the four endings deterministically per row (updates gold index accordingly). |

## Scorers

| name | inputs | behaviour |
| --- | --- | --- |
| `multiple_choice_letter` | numeric label (0-based) or letter label, list of choice texts | Extracts the first standalone `A`/`B`/`C`/`D` from the model output. Falls back to the choice text whose words appear first in the output. |
| `gsm8k_final` | gold answer string | Reads the last `#### <number>` line (GSM8K convention); else the last number in the output. |
| `exact_normalize` | gold answer string | Lowercases, strips punctuation and whitespace, then exact match. |

## Experiment artifacts

Runs write under `<repo>/data/exps/` by default (`SOPHON_BENCH_OUT_DIR` overrides). Each invocation creates `<out>/<run_id>/experiment_summary.json` plus `<out>/<run_id>/<task_id>/predictions.jsonl` and `summary.json`. See `sophon/docs/README.md` for `run_id` composition and CLI flags.

## Adding a new task

1. Drop the data file under `data/<task>/...` (JSONL preferred).
2. Add `data/benchmarks/<task>.yaml` referencing it.
3. Run `uv run sophon-benchmark --task <task> --limit 5` to sanity-check.
