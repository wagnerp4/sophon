# Energy regime (`/energy`)

`/mode plan|chat|agent` is harness permission. `/energy local|api` is which scarce resource the session is allowed to burn. They are orthogonal. Do not overload `/mode`.

Backends already exist as a picker (`/backend`, `/model openai:NAME`). Energy is the regime those pickers must obey, plus a spend ledger for API.

Claude Code the **CLI** stays a non-goal ([harness.md](harness.md)). API mode talks to hosted model HTTP APIs (OpenAI, Anthropic Messages, Google Gemini, DeepSeek). It does not spawn `claude` or `codex` as a child runtime.

## Two regimes

| Regime | Scarce resource | Legal backends | Default child model |
| --- | --- | --- | --- |
| `local` | 24G VRAM, one compute stream | `hf`, `ollama`, `lmstudio` | inherit resident weights |
| `api` | dollars, RPM, TPM | `openai`, `anthropic`, `google`, later `deepseek` | inherit parent route |

This is the same split as [orchestrator.md](orchestrator.md) and [subagents.md](subagents.md). The new piece is a **session-level swap** with spend controls, not only a placement table for children.

`hybrid` (local parent, API children or the reverse) stays a later placement flag. v1 of `/energy` is a hard swap of the interactive session.

## Current behavior

- `/backend auto|ollama|lmstudio|hf|openai|anthropic|google` and `/model openai:NAME` switch the chat backend.
- `ProviderSpec.is_local` already distinguishes local vs hosted ([src/backend/providers.py](../../src/backend/providers.py)).
- DeepSeek HTTP is on the [TODO.md](../TODO.md) model list. It is not a `ChatBackendId` yet.
- No daily/monthly USD cap. No degrade-to-local. No unload of the 12B when switching to API.
- Subagent v1 inherits whichever parent is resident. `maxActive=1` is a GPU rule. It should not apply as a hard cap on API children once energy is `api` (API live-child cap is `SOPHON_SWARM_MAX_API` / a later `SOPHON_ENERGY_API_MAX_ACTIVE`).

## Intended slash surface

```text
/energy                 show regime, backend, spend today / window, caps
/energy local           restore last local backend. Picker only if that is missing.
/energy api             follow-up dialogue: pick openai / anthropic / google, then backend swap
/energy api openai      skip the provider step. Model picker if that provider has no last model
/energy cap             show caps
/energy cap daily 5     USD per UTC day (0 = off)
/energy cap monthly 40  USD per UTC month
/energy cap rpm 60      requests per minute across API providers
/energy on-limit ask    ask and hold: 1 continue this request, 3 stop. Timeout stops.
/energy persist         write .sophon/energy.local.yaml
```

Bare `/energy api` does not switch until a provider is chosen. Esc or an empty REPL line cancels. Local weights stay loaded. `/backend` remains a direct switch and still unloads. `/energy` does not.

Session-only until `/energy persist` writes `.sophon/energy.local.yaml` (gitignored). Env defaults:

```powershell
$env:SOPHON_ENERGY="local"
$env:SOPHON_ENERGY_DAILY_USD="5"
$env:SOPHON_ENERGY_MONTHLY_USD="40"
$env:SOPHON_ENERGY_RPM="60"
$env:SOPHON_ENERGY_ON_LIMIT="ask"
$env:SOPHON_ENERGY_ASK_TIMEOUT_S="60"
```

`SOPHON_ENERGY=api` is valid when at least one hosted key is present. Otherwise boot in `local` and print the missing key hint.

## Spend ledger

Append-only JSONL under `data/energy/spend.jsonl` (or `~/.cache/sophon/energy/` when the checkout is on `/mnt/c` from WSL, same pattern as chat logs).

Each row: UTC ts, provider, model, input tokens, output tokens, cached tokens if known, USD estimate, request id, session id. Secrets never land in the row.

Price table: checked-in `src/backend/energy/prices.yaml` with a dated snapshot. Unknown model: use that provider’s default chat rate and mark `estimate=rough`. Do not call a billing HTTP API in v1.

Chat HUD: `energy=api $1.20/$5`. Soft warn at 80% of the daily cap (one line per session).

Hard stop happens **before** the next API HTTP call, not after a 429. Soft warn at 80% of the daily cap (one chat line per session).

## Switch behavior

`/energy api` is a follow-up selection, then a backend swap. It does not unload HF, LM Studio, or Ollama.

1. Record the current local combo as `last_local` when leaving local.
2. Prompt for a hosted backend that has a key. Disabled rows explain the missing key.
3. If that provider has `last_api`, bind it. Else open the model picker filtered to that group.
4. Refuse a provider with no key. Cancel leaves the current backend.

`/energy local` restores `last_local`. If that is empty, prompt among `hf`, `ollama`, `lmstudio`. Spend ledger stays. Caps do not apply to local tokens.

On limit (`ask`): hold the turn, prompt 1 continue / 3 stop. If `SOPHON_ENERGY_ASK_TIMEOUT_S` elapses, or there is no TTY answer, stop. Do not start the HTTP call.

Failed API call (401, 429, insufficient quota) honors `on-limit`. 429 is not a spend-cap hit. It is a provider throttle. Backoff once, then apply `on-limit`.

## Implementation sketch

- `src/backend/energy/regime.py` - `EnergyRegime`, load/save, `assert_api_allowed(estimated_usd)`.
- `src/backend/energy/ledger.py` - append JSONL, sum today / month.
- `src/backend/energy/prices.yaml` - static rates.
- Slash in `src/cli/chat.py` next to `/backend`. `/help` section `energy` or under `model`.
- Chat status bar: `energy=api $1.20/$5`.
- `managed_backend_ids()` grows `deepseek` when that provider ships. DeepSeek is an API-regime backend, not a third energy value.
- Subagent admission reads `regime`. Local: `maxActiveLocal=1`. API: inherit parent route, cap by RPM/USD.

TODO: DeepSeek `ChatBackendId` + `SOPHON_DEEPSEEK_API_KEY`.
TODO: Prompt-cache pricing once Anthropic/OpenAI expose it on the usage object we already parse.
TODO: Per-provider caps (`/energy cap openai daily 5`). v1 is one global USD cap.

## Non-goals

- Wrapping the Claude Code or Codex CLIs as “API mode”.
- Auto-switching mid-turn because VRAM spiked.
- Treating LM Studio cloud or OpenRouter as a hidden fourth regime. If added, they are `api` providers with their own price rows.
- Billing dashboards beyond the HUD line and `/energy`.

## References

- [OpenAI usage in chat completions](https://platform.openai.com/docs/api-reference/chat/object)
- [Anthropic Messages usage](https://docs.anthropic.com/en/api/messages)
- [DeepSeek API docs](https://api-docs.deepseek.com/)
- [OpenAI pricing](https://openai.com/api/pricing/)
- [Anthropic pricing](https://www.anthropic.com/pricing)
- [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing)
