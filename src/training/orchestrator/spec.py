from __future__ import annotations

from dataclasses import dataclass, replace


DRIVER_HF_LORA = "hf-lora"
DRIVER_SSL4SED = "ssl4sed"
DRIVERS = (DRIVER_HF_LORA, DRIVER_SSL4SED)


@dataclass
class TrainerSpec:
    driver: str = DRIVER_HF_LORA
    task: str = ""
    data: str = ""
    model: str = ""
    target: str = ""
    fast_dev: bool = True
    overrides: tuple[str, ...] = ()


def spec_from_session(state: object) -> TrainerSpec:
    driver = str(getattr(state, "trainer_driver", "") or DRIVER_HF_LORA)
    if driver not in DRIVERS:
        driver = DRIVER_HF_LORA
    return TrainerSpec(
        driver=driver,
        task=str(getattr(state, "trainer_task", "") or ""),
        data=str(getattr(state, "trainer_data", "") or ""),
        model=str(getattr(state, "trainer_model", "") or ""),
        target=str(getattr(state, "trainer_target", "") or ""),
        fast_dev=bool(getattr(state, "trainer_fast_dev", True)),
    )


def spec_to_session(state: object, spec: TrainerSpec) -> None:
    state.trainer_driver = spec.driver
    state.trainer_task = spec.task
    state.trainer_data = spec.data
    state.trainer_model = spec.model
    state.trainer_target = spec.target
    state.trainer_fast_dev = spec.fast_dev


def merge_select(spec: TrainerSpec, fields: dict[str, str]) -> TrainerSpec:
    task = fields.get("task", spec.task)
    data = fields.get("data", spec.data)
    model = fields.get("model", spec.model)
    target = fields.get("target", spec.target)
    return replace(spec, task=task, data=data, model=model, target=target)
