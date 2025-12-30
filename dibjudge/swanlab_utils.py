from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import warnings


def _sanitize_config(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _sanitize_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_config(v) for v in value]
    return str(value)


def init_swanlab(
    enabled: bool,
    project: Optional[str],
    run_name: Optional[str],
    config: Optional[Dict[str, Any]],
    log_dir: Optional[str],
    tags: Optional[Sequence[str]],
):
    if not enabled:
        return None
    try:
        import swanlab
    except ImportError:
        warnings.warn("swanlab is not installed; skipping logging.")
        return None
    warnings.filterwarnings(
        "ignore", category=ResourceWarning, module=r"swanlab\\..*"
    )

    kwargs: Dict[str, Any] = {}
    if project:
        kwargs["project"] = project
    if run_name:
        kwargs["name"] = run_name
    if config:
        kwargs["config"] = _sanitize_config(config)
    if log_dir:
        kwargs["dir"] = log_dir
    if tags:
        kwargs["tags"] = list(tags)

    try:
        swanlab.init(**kwargs)
    except TypeError:
        kwargs.pop("dir", None)
        kwargs.pop("tags", None)
        swanlab.init(**kwargs)

    return swanlab


def log_swanlab(client, metrics: Dict[str, float], step: Optional[int] = None) -> None:
    if client is None:
        return
    if step is None:
        client.log(metrics)
        return
    try:
        client.log(metrics, step=step)
    except TypeError:
        client.log(metrics)


def finish_swanlab(client) -> None:
    if client is None:
        return
    finish = getattr(client, "finish", None)
    if callable(finish):
        finish()
