from __future__ import annotations

from pathlib import Path

from backend.hf.registry import (
    HF_MODEL_PRESETS,
    finetune_eligible_keys,
    model_dir_has_complete_weights,
    resolve_preset_dir,
)
from training.finetune.datasets.load import dataset_source_summary
from training.finetune.datasets.registry import FINETUNE_DATASET_PRESETS, match_dataset_preset
from training.orchestrator.protocol import CatalogRow, LogFn
from training.orchestrator.spec import DRIVER_HF_LORA, TrainerSpec
from utils.device.env_bootstrap import sophon_project_root


def local_dataset_ready(key: str, project_root: Path) -> bool:
    preset = FINETUNE_DATASET_PRESETS.get(key)
    if preset is None:
        return False
    if preset.hub_id:
        return True
    if preset.local_kind == "mmlu_csv":
        from training.finetune.datasets.load import _mmlu_rows

        try:
            _mmlu_rows(project_root, preset.split)
            return True
        except FileNotFoundError:
            return False
    summary = dataset_source_summary(preset, project_root)
    return "missing" not in summary


def session_trainable(state: object) -> tuple[bool, str]:
    backend = str(getattr(state, "backend_id", "") or "")
    preset_key = getattr(state, "preset_key", None)
    root = getattr(state, "project_root", None) or sophon_project_root()
    if backend != "hf":
        return False, f"backend={backend} is not AutoModelForCausalLM"
    if not preset_key:
        return False, "no HF preset selected"
    preset = HF_MODEL_PRESETS.get(str(preset_key))
    if preset is None:
        return False, f"unknown preset {preset_key!r}"
    if not preset.finetune_eligible:
        reason = preset.finetune_reason or "not a causal LoRA SFT target"
        return False, reason
    model_dir = resolve_preset_dir(str(preset_key), root)
    if not (model_dir / "config.json").is_file() or not model_dir_has_complete_weights(model_dir):
        return False, f"weights missing at {model_dir}"
    return True, "hf causal LoRA"


def nearest_eligible_preset(project_root: Path) -> tuple[str | None, bool]:
    for key in finetune_eligible_keys(include_large=False):
        model_dir = resolve_preset_dir(key, project_root)
        on_disk = (model_dir / "config.json").is_file() and model_dir_has_complete_weights(model_dir)
        if on_disk:
            return key, True
    keys = finetune_eligible_keys(include_large=False)
    if keys:
        return keys[0], False
    return None, False


class HfLoraDriver:
    driver_id = DRIVER_HF_LORA

    def __init__(self, state: object) -> None:
        self.state = state
        self.project_root = getattr(state, "project_root", None) or sophon_project_root()

    def inventory(self) -> list[CatalogRow]:
        rows: list[CatalogRow] = [
            CatalogRow("task", "instruction-sft", "ready", "Alpaca SFT via hf_peft/unsloth"),
        ]
        for key, preset in FINETUNE_DATASET_PRESETS.items():
            ready = local_dataset_ready(key, self.project_root)
            summary = dataset_source_summary(preset, self.project_root)
            rows.append(
                CatalogRow(
                    "data",
                    key,
                    "ready" if ready else "missing",
                    summary,
                )
            )
        for key in finetune_eligible_keys(include_large=True):
            preset = HF_MODEL_PRESETS[key]
            model_dir = resolve_preset_dir(key, self.project_root)
            on_disk = (model_dir / "config.json").is_file() and model_dir_has_complete_weights(model_dir)
            status = "ready" if on_disk else "missing"
            detail = preset.repo_id
            if preset.finetune_requires_allow_large:
                detail = detail + " allow_large"
            rows.append(CatalogRow("model", key, status, detail, preset.vram_class))
        return rows

    def preflight(self, spec: TrainerSpec) -> list[str]:
        lines: list[str] = []
        data = spec.data or "gsm8k_instructions"
        matched = match_dataset_preset(data)
        if matched is None:
            return [f"unknown dataset {data!r}"]
        if not local_dataset_ready(matched, self.project_root):
            lines.append(f"data {matched} missing on disk")
        model = spec.model or str(getattr(self.state, "preset_key", "") or "")
        if not model:
            return lines + ["no model selected (set /trainer select model=PRESET or /model PRESET)"]
        preset = HF_MODEL_PRESETS.get(model)
        if preset is None:
            return lines + [f"unknown preset {model!r}"]
        if not preset.finetune_eligible:
            return lines + [f"ineligible: {preset.finetune_reason or model}"]
        model_dir = resolve_preset_dir(model, self.project_root)
        if not (model_dir / "config.json").is_file() or not model_dir_has_complete_weights(model_dir):
            lines.append(f"weights missing. /model-download {model}")
        return lines

    def fetch_data(self, key: str, on_log: LogFn) -> None:
        matched = match_dataset_preset(key)
        if matched is None:
            on_log(f"unknown dataset {key!r}")
            return
        preset = FINETUNE_DATASET_PRESETS[matched]
        if preset.hub_id:
            on_log(f"{matched}: Hub {preset.hub_id} is fetched when /trainer run starts")
            return
        summary = dataset_source_summary(preset, self.project_root)
        on_log(f"{matched}: {summary}")
        if "missing" in summary:
            on_log("place the local split under data/ then /trainer data")

    def run(self, spec: TrainerSpec, on_log: LogFn) -> None:
        from cli.chat import _record_train_progress, _unload_model_weights
        from training.finetune.job import run_finetune_job

        dataset_id = spec.data or "gsm8k_instructions"
        preset_key = spec.model or getattr(self.state, "preset_key", None)
        if not preset_key:
            raise ValueError("hf-lora run needs model=PRESET or a selected HF preset")
        _unload_model_weights(self.state)
        self.state.finetune_running = True
        self.state.train_loss_history = []
        try:
            on_log(
                f"starting finetune preset={preset_key!r} dataset={dataset_id!r}"
            )
            result = run_finetune_job(
                str(preset_key),
                dataset_id,
                project_root=self.project_root,
                backend_id="auto",
                recipe_path=None,
                recipe_override_tokens=list(spec.overrides),
                on_progress=lambda step, total, label: _record_train_progress(
                    self.state, step, total, label
                ),
                on_log=on_log,
            )
        finally:
            self.state.finetune_running = False
        self.state.preset_key = result.meta.preset_key
        self.state.last_finetune_run_dir = result.run_dir
        self.state.last_finetune_adapter_name = result.adapter_name
        self.state.adapter_path = str(result.adapter_dir.resolve())
        on_log(f"(finetune done; adapter at {result.adapter_dir})")
        load_name = result.adapter_name or str(result.adapter_dir)
        on_log(f"(backend={result.backend_id}; /adapter load {load_name})")

    def status(self) -> list[str]:
        lines: list[str] = []
        if getattr(self.state, "finetune_running", False):
            lines.append("hf-lora job running")
        run_dir = getattr(self.state, "last_finetune_run_dir", None)
        name = getattr(self.state, "last_finetune_adapter_name", None)
        if run_dir is not None:
            lines.append(f"last run_dir={run_dir}")
        if name:
            lines.append(f"last adapter={name}")
        if not lines:
            lines.append("no hf-lora run in this session")
        return lines

    def stop(self) -> list[str]:
        # TODO: cooperative cancel for run_finetune_job on the chat turn worker
        return ["hf-lora stop is not wired (job runs on the turn worker)"]
