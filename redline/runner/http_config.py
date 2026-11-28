"""Set up a web app target from a YAML file instead of Python code."""

from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import ConfigDict, Field, ValidationError

from redline.core import RedlineModel
from redline.detectors import JudgeConfig
from redline.runner.app_factory import HttpFactory
from redline.targets import HttpTemplateConfig, RateLimiter


class ConfigError(Exception):
    """A target config file is missing or wrong. The message never quotes its values."""


class HttpAppConfig(RedlineModel):
    """The whole file: how to call the app, plus how to judge and pace the attacks."""

    # Keeps rejected values (possibly real keys) out of error messages.
    model_config = ConfigDict(hide_input_in_errors=True)

    target: HttpTemplateConfig
    canary: str = Field(min_length=1)
    tool_names: tuple[str, ...] = ()
    leak_phrases: tuple[str, ...] = ()
    max_calls_per_attempt: int | None = Field(default=None, ge=1)
    retry_rate_limits: bool = False
    safe_to_retry: bool = False
    requests_per_minute: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    # Optional LLM judge (`judge: {}` takes the local defaults) and your app's system prompt.
    judge: JudgeConfig | None = None
    judge_system_prompt: str | None = None


class _NoDuplicatesLoader(yaml.SafeLoader):
    """SafeLoader that refuses a repeated key. Plain YAML silently keeps the last one."""


def _mapping(loader: _NoDuplicatesLoader, node: yaml.MappingNode) -> dict[Any, Any]:
    keys = [key.value for key, _ in node.value if isinstance(key, yaml.ScalarNode)]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        line = node.start_mark.line + 1
        raise ConfigError(f"{', '.join(repeated)} appears twice in the block at line {line}")
    return loader.construct_mapping(node, deep=True)


_NoDuplicatesLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _read(path: Path) -> HttpAppConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(f"can't read {path}: {type(error).__name__}") from None
    try:
        # A SafeLoader subclass, so only plain data can be built. S506 can't see that.
        data = yaml.load(text, Loader=_NoDuplicatesLoader)  # noqa: S506
    except yaml.MarkedYAMLError as error:
        # Not str(error): it quotes the offending line, which may hold a key. For the same
        # reason `from None` drops the original from any traceback.
        mark = error.problem_mark
        where = f" at line {mark.line + 1}" if mark is not None else ""
        raise ConfigError(f"{path.name}: invalid YAML{where}: {error.problem}") from None
    except yaml.YAMLError:
        raise ConfigError(f"{path.name}: invalid YAML") from None
    try:
        return HttpAppConfig.model_validate(data)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '(whole file)'}: {e['msg']}"
            for e in error.errors()
        )
        raise ConfigError(f"{path.name}: {problems}") from None
    except TypeError:
        # E.g. an unquoted date in the body: YAML makes it a date, which JSON can't carry.
        raise ConfigError(
            f"{path.name}: the body has a value that can't be sent as JSON; put it in quotes"
        ) from None


def load_http_factory(path: Path, *, client: httpx.AsyncClient | None = None) -> HttpFactory:
    """Build an `HttpFactory` from a YAML file. Raises ConfigError if anything is wrong."""
    config = _read(path)
    limiter = (
        None if config.requests_per_minute is None else RateLimiter(config.requests_per_minute)
    )
    try:
        return HttpFactory(
            config.target,
            canary=config.canary,
            tool_names=config.tool_names,
            leak_phrases=config.leak_phrases,
            max_calls_per_attempt=config.max_calls_per_attempt,
            retry_rate_limits=config.retry_rate_limits,
            safe_to_retry=config.safe_to_retry,
            client=client,
            limiter=limiter,
            judge=config.judge,
            judge_system_prompt=config.judge_system_prompt,
        )
    except ValueError as error:
        raise ConfigError(f"{path.name}: {error}") from None
