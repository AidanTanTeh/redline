"""Adapter for any web app with a JSON endpoint, described by a request template."""

import json
import re
import time
from typing import Any, Literal, Self

import httpx
from pydantic import ConfigDict, Field, model_validator

from redline.core import RedlineModel, TargetResponse, ToolCall
from redline.targets.limiter import RateLimiter
from redline.targets.secrets import is_safe_header_text, read_secret, redact

PLACEHOLDER = "{prompt}"
# ${NAME} in a header value is replaced by that environment variable at send time.
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# A header whose name CONTAINS one of these must use ${ENV_NAME}.
_SECRET_HEADER_HINTS = (
    "auth",
    "cookie",
    "key",
    "token",
    "secret",
    "session",
    "password",
    "passwd",
    "credential",
)
# Query and body names must match one of these EXACTLY (ignoring case, "-" and "_"). Matching
# parts of names would reject ordinary fields such as "max_tokens" or "session_id".
_SECRET_FIELD_NAMES = frozenset(
    {
        "accesstoken",
        "apikey",
        "auth",
        "authorization",
        "authtoken",
        "bearertoken",
        "clientsecret",
        "key",
        "passwd",
        "password",
        "privatekey",
        "refreshtoken",
        "secret",
        "sessiontoken",
        "token",
    }
)
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _names_a_secret(name: str) -> bool:
    return name.casefold().replace("-", "").replace("_", "") in _SECRET_FIELD_NAMES


def _secret_fields(body: Any) -> list[str]:
    """Names of body fields, at any depth, that hold a fixed text value and sound secret."""
    if isinstance(body, dict):
        found = [
            str(key)
            for key, value in body.items()
            if isinstance(value, str) and PLACEHOLDER not in value and _names_a_secret(str(key))
        ]
        return found + [name for value in body.values() for name in _secret_fields(value)]
    if isinstance(body, list | tuple):
        return [name for value in body for name in _secret_fields(value)]
    return []


class HttpTemplateConfig(RedlineModel):
    """How to call one endpoint and read its reply. Saved in reports: put no secrets in it."""

    # Pydantic errors normally print the rejected input, which here could be a real key.
    model_config = ConfigDict(hide_input_in_errors=True)

    url: str = Field(min_length=1)
    method: Literal["POST", "PUT", "PATCH"] = "POST"
    # Header values may use ${ENV_NAME}. Secret headers (Authorization, API keys) must.
    headers: dict[str, str] = Field(default_factory=dict)
    # The JSON body. "{prompt}" inside any text value is replaced by the attack's message.
    body: dict[str, Any]
    # Where the reply text is in the response JSON, e.g. "data.reply" or "choices.0.text".
    response_path: str = Field(min_length=1)
    # Where the list of tool calls is, if the app reports them. Without this, a tool call
    # can't be seen, so tool attacks must not be run against the target.
    tool_calls_path: str | None = Field(default=None, min_length=1)
    tool_name_key: str = "name"
    tool_arguments_key: str = "arguments"
    # Where the app says why it stopped. Without this, a cut-off or blocked reply counts as a
    # normal answer, so an attack on it looks "resisted".
    finish_reason_path: str | None = Field(default=None, min_length=1)
    # The app's own values for "cut off" and "blocked", ignoring case (e.g. "MAX_TOKENS").
    cut_off_values: tuple[str, ...] = ("length",)
    blocked_values: tuple[str, ...] = ("content_filter",)
    model: str = Field(default="unknown", min_length=1)
    timeout_s: float = Field(default=60.0, gt=0)
    # Set True to say you own this system or have permission to test it. Needed for any
    # host other than your own machine. A reminder, not a security control.
    authorized: bool = False

    @model_validator(mode="after")
    def _url_is_safe_to_attack_and_to_save(self) -> Self:
        url = httpx.URL(self.url)  # parsed by the same code that will send the request
        if url.scheme not in ("http", "https") or not url.host:
            raise ValueError("url must be a full http or https address")
        if url.userinfo:
            raise ValueError("don't put a username or password in the url; use a header")
        if url.host not in _LOCAL_HOSTS and not self.authorized:
            raise ValueError(
                f"{url.host} is not this machine. Only test systems you own or are "
                "authorised to test, and then set authorized: true"
            )
        return self

    @model_validator(mode="after")
    def _secret_headers_come_from_the_environment(self) -> Self:
        for name, value in self.headers.items():
            # E.g. "Bearer ${KEY} " with a stray space would be rejected at send time, in an
            # error quoting the whole header, key included.
            if not is_safe_header_text(value):
                raise ValueError(f"header {name} has spaces at the ends or unusual characters")
            secret = any(hint in name.casefold() for hint in _SECRET_HEADER_HINTS)
            if secret and not _ENV_REFERENCE.search(value):
                # Name the header only: the value may be a real key.
                raise ValueError(f"header {name} must use ${{ENV_NAME}}, not a literal value")
        return self

    @model_validator(mode="after")
    def _no_literal_secret_in_the_url_or_body(self) -> Self:
        # Neither supports ${ENV_NAME}, so a secret has to move to a header.
        found = [name for name in httpx.URL(self.url).params if _names_a_secret(name)]
        found += _secret_fields(self.body)
        if found:
            raise ValueError(
                f"{', '.join(found)} looks like a secret in the url or body, which are saved "
                "in reports. Send it in a header as ${ENV_NAME}"
            )
        return self

    @model_validator(mode="after")
    def _finish_values_need_a_path(self) -> Self:
        given = self.model_fields_set & {"cut_off_values", "blocked_values"}
        if given and self.finish_reason_path is None:
            raise ValueError("cut_off_values and blocked_values need finish_reason_path")
        return self

    @model_validator(mode="after")
    def _body_uses_the_prompt(self) -> Self:
        # Without it every attack would send the same fixed request and "resist".
        if PLACEHOLDER not in json.dumps(self.body):
            raise ValueError(f"body must contain {PLACEHOLDER} in one of its text values")
        return self

    @property
    def env_names(self) -> tuple[str, ...]:
        """The environment variables the headers need, so a run can check them up front."""
        names = (n for value in self.headers.values() for n in _ENV_REFERENCE.findall(value))
        return tuple(sorted(set(names)))


def _fill(template: Any, prompt: str) -> Any:
    """A copy of the body with the prompt put in. Done on the data, not on JSON text, so
    quotes or braces in an attack can't change the shape of the request."""
    if isinstance(template, str):
        return template.replace(PLACEHOLDER, prompt)
    if isinstance(template, dict):
        return {key: _fill(value, prompt) for key, value in template.items()}
    if isinstance(template, list | tuple):
        return [_fill(value, prompt) for value in template]
    return template


def _lookup(data: Any, path: str) -> Any:
    """Follow a dotted path into parsed JSON; numbers index into lists."""
    for part in path.split("."):
        if isinstance(data, dict) and part in data:
            data = data[part]
        elif isinstance(data, list) and part.isdecimal() and int(part) < len(data):
            data = data[int(part)]
        else:
            raise ValueError(f"the response has nothing at '{path}'")
    return data


def _header_value(template: str) -> str:
    return _ENV_REFERENCE.sub(lambda match: read_secret(match.group(1)), template)


class HttpTemplateTarget:
    """A Target that sends each prompt to a JSON endpoint."""

    def __init__(
        self,
        config: HttpTemplateConfig,
        client: httpx.AsyncClient | None = None,
        *,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._config = config
        self._client = client
        self._limiter = limiter

    @property
    def name(self) -> str:
        url = httpx.URL(self._config.url)
        return f"http/{url.host}{url.path}"

    async def send(self, prompt: str) -> TargetResponse:
        config = self._config
        headers = {name: _header_value(value) for name, value in config.headers.items()}
        body = _fill(config.body, prompt)

        if self._limiter is not None:
            await self._limiter.wait()
        start = time.perf_counter()
        if self._client is not None:
            response = await self._request(self._client, headers, body)
        else:
            async with httpx.AsyncClient() as client:
                response = await self._request(client, headers, body)
        latency_ms = (time.perf_counter() - start) * 1000

        if not response.is_success:
            # Not raise_for_status(): its message includes the full url, which is saved with
            # the results and may carry something private.
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code} from {response.request.url.host}",
                request=response.request,
                response=response,
            )
        data = response.json()
        tool_calls = self._tool_calls(data)
        finish_reason = self._finish_reason(data)
        text = _lookup(data, config.response_path)
        # Many APIs send null text on a turn that only calls tools, or that was stopped.
        if text is None and (tool_calls or finish_reason in ("length", "content_filter")):
            text = ""
        if not isinstance(text, str):
            raise ValueError(f"'{config.response_path}' is not text")
        reply = TargetResponse(
            text=text,
            model=config.model,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )
        # Every environment value that went into a header.
        return redact(reply, [read_secret(name) for name in config.env_names])

    def _finish_reason(self, data: Any) -> str | None:
        """The app's stop reason, with its cut-off and blocked values translated into the
        two words the runner counts as "no answer"."""
        config = self._config
        if config.finish_reason_path is None:
            return None
        raw = _lookup(data, config.finish_reason_path)
        if raw is None:
            return None
        if not isinstance(raw, str):
            raise ValueError(f"'{config.finish_reason_path}' is not text")
        value = raw.casefold()
        if value in (v.casefold() for v in config.cut_off_values):
            return "length"
        if value in (v.casefold() for v in config.blocked_values):
            return "content_filter"
        return raw

    async def _request(
        self, client: httpx.AsyncClient, headers: dict[str, str], body: Any
    ) -> httpx.Response:
        # No redirects: one could forward the attack to a host that was never authorised.
        return await client.request(
            self._config.method,
            self._config.url,
            json=body,
            headers=headers,
            timeout=self._config.timeout_s,
            follow_redirects=False,
        )

    def _tool_calls(self, data: Any) -> tuple[ToolCall, ...]:
        config = self._config
        if config.tool_calls_path is None:
            return ()
        # A missing path is an error: a typo would otherwise hide every tool call.
        raw = _lookup(data, config.tool_calls_path)
        if raw is None:
            return ()
        if not isinstance(raw, list):
            raise ValueError(f"'{config.tool_calls_path}' is not a list")
        calls = []
        for item in raw:
            name = _lookup(item, config.tool_name_key)
            if not isinstance(name, str) or not name:
                raise ValueError("a tool call has no name")
            arguments = _lookup(item, config.tool_arguments_key)
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            if not isinstance(arguments, dict):
                arguments = {"_raw": arguments}
            calls.append(ToolCall(name=name, arguments=arguments))
        return tuple(calls)
