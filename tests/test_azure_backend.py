"""Azure Foundry adapter, offline: client construction and reply translation (no network calls)."""

from types import SimpleNamespace

import httpx2 as httpx
import openai
import pytest

from problem_clustering.backend import ContentFilteredError
from problem_clustering.backend.azure import AzureFoundryBackend, FoundrySettings
from problem_clustering.config import BackendConfig

ENDPOINT = "https://my-resource.openai.azure.com/"
MSG = [{"role": "user", "content": "x"}]


def settings(endpoint=ENDPOINT):
    return FoundrySettings(api_key="k", endpoint=endpoint, _env_file=None)


def make(stub_reply=None, raises=None):
    backend = AzureFoundryBackend(BackendConfig(), settings())

    def create(**kwargs):
        if raises is not None:
            raise raises
        return stub_reply

    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return backend


def completion(content=None, finish_reason="stop", refusal=None):
    msg = SimpleNamespace(content=content, tool_calls=None, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
                           usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))


def test_sdk_retries_are_off_so_only_the_runtime_retries():
    assert AzureFoundryBackend(BackendConfig(), settings()).client.max_retries == 0


@pytest.mark.parametrize("endpoint", [
    "https://my-resource.openai.azure.com/",
    "https://my-resource.cognitiveservices.azure.com",
    "https://my-project.services.ai.azure.com/",
])
def test_resource_endpoints_are_accepted(endpoint):
    assert settings(endpoint).endpoint == endpoint


@pytest.mark.parametrize("endpoint", [
    "my-resource.openai.azure.com",  # no scheme
    "http://my-resource.openai.azure.com",  # not https
    "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions",  # full request URL
])
def test_malformed_endpoints_fail_with_a_clear_message(endpoint):
    with pytest.raises(ValueError, match="FOUNDRY_ENDPOINT"):
        settings(endpoint)


def test_unknown_endpoint_host_warns():
    with pytest.warns(UserWarning, match="FOUNDRY_ENDPOINT"):
        settings("https://example.com")


def test_plain_reply_is_translated():
    reply = make(completion(content='{"a": 1}')).chat(MSG)
    assert reply.json() == {"a": 1} and reply.usage.prompt_tokens == 3


@pytest.mark.parametrize("stub", [
    completion(finish_reason="content_filter"),
    completion(refusal="I can't help with that."),
])
def test_filtered_or_refused_reply_raises_content_filtered_error(stub):
    with pytest.raises(ContentFilteredError):
        make(stub).chat(MSG)


def test_filtered_prompt_400_raises_content_filtered_error():
    response = httpx.Response(400, request=httpx.Request("POST", ENDPOINT))
    exc = openai.BadRequestError("filtered", response=response, body={"code": "content_filter"})
    with pytest.raises(ContentFilteredError):
        make(raises=exc).chat(MSG)


def test_other_400_is_not_a_content_filter():
    response = httpx.Response(400, request=httpx.Request("POST", ENDPOINT))
    exc = openai.BadRequestError("bad", response=response, body={"code": "invalid_request"})
    with pytest.raises(openai.BadRequestError):
        make(raises=exc).chat(MSG)
