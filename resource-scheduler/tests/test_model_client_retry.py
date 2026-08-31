"""
Tests for ModelClient's retry/backoff behavior -- no live API needed.
Fakes the underlying OpenAI client so retryable and non-retryable
failures can be exercised deterministically. This is the fix for the
504 Gateway Time-out that hit every live Task Prioritization call this
session; the point of this test is to prove the retry logic itself is
correct without needing a flaky live endpoint to reproduce it.
"""
import openai
import pytest

from resource_scheduler.model_client import ModelClient, _RETRYABLE_EXCEPTIONS


def _fake_response(content="{}"):
    class Message:
        def __init__(self):
            self.content = content
            self.tool_calls = None
            self.reasoning = None

    class Choice:
        def __init__(self):
            self.message = Message()

    class Response:
        def __init__(self):
            self.choices = [Choice()]
            self.usage = None

    return Response()


class _FakeRequest:
    """Stand-in for httpx.Request -- openai's error constructors just
    store this and never inspect it further (confirmed by reading
    APIError/APIConnectionError/APITimeoutError's actual __init__
    source). Deliberately not depending on a real httpx/httpx2 install:
    this environment happened to resolve openai's http dependency as a
    vendored `httpx2` fork rather than plain `httpx`, which is not
    something a test should hard-code -- a normal `pip install openai`
    elsewhere would pull in real httpx instead."""
    pass


class _FakeResponse:
    """Stand-in for httpx.Response -- APIStatusError.__init__ only reads
    .request, .status_code, and .headers.get(...) (confirmed the same
    way), never checks the type, so this is enough."""
    def __init__(self, status_code: int):
        self.request = _FakeRequest()
        self.status_code = status_code
        self.headers: dict[str, str] = {}


def _fake_error(cls):
    """Builds a real instance of an openai exception class with the
    minimal args each constructor actually requires -- these aren't
    plain Exception subclasses, they need a message/response/body."""
    if cls is openai.RateLimitError or cls is openai.InternalServerError:
        response = _FakeResponse(500 if cls is openai.InternalServerError else 429)
        return cls("boom", response=response, body=None)
    if cls is openai.APITimeoutError:
        return cls(request=_FakeRequest())
    if cls is openai.APIConnectionError:
        return cls(message="boom", request=_FakeRequest())
    raise ValueError(cls)


class _FakeCompletions:
    def __init__(self, side_effects):
        self.side_effects = list(side_effects)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        effect = self.side_effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect


def _make_client_with_fake(side_effects, **client_kwargs) -> tuple[ModelClient, _FakeCompletions]:
    client = ModelClient(base_url="https://example.test/v1", api_key="fake", default_model="fake-model", **client_kwargs)
    fake_completions = _FakeCompletions(side_effects)

    class FakeChat:
        completions = fake_completions

    client._client.chat = FakeChat()
    return client, fake_completions


def test_succeeds_immediately_with_no_retry_needed(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client, fake = _make_client_with_fake([_fake_response("hello")])
    response = client.call([{"role": "user", "content": "hi"}])
    assert response.text == "hello"
    assert fake.calls == 1


@pytest.mark.parametrize("exc_cls", [openai.InternalServerError, openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError])
def test_retries_on_each_retryable_exception_type(monkeypatch, exc_cls):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    client, fake = _make_client_with_fake([_fake_error(exc_cls), _fake_response("recovered")], max_retries=3)
    response = client.call([{"role": "user", "content": "hi"}])
    assert response.text == "recovered"
    assert fake.calls == 2
    assert len(sleeps) == 1


def test_backoff_doubles_each_attempt(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    client, fake = _make_client_with_fake(
        [_fake_error(openai.InternalServerError), _fake_error(openai.InternalServerError), _fake_response("ok")],
        max_retries=3, retry_backoff_seconds=2.0,
    )
    response = client.call([{"role": "user", "content": "hi"}])
    assert response.text == "ok"
    assert sleeps == [2.0, 4.0]


def test_raises_after_exhausting_all_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    errors = [_fake_error(openai.InternalServerError) for _ in range(4)]  # max_retries=3 -> 4 total attempts
    client, fake = _make_client_with_fake(errors, max_retries=3)
    with pytest.raises(openai.InternalServerError):
        client.call([{"role": "user", "content": "hi"}])
    assert fake.calls == 4


def test_does_not_retry_non_retryable_errors(monkeypatch):
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    response = _FakeResponse(400)
    client, fake = _make_client_with_fake([openai.BadRequestError("bad request", response=response, body=None)])
    with pytest.raises(openai.BadRequestError):
        client.call([{"role": "user", "content": "hi"}])
    assert fake.calls == 1
    assert sleeps == []


def test_on_retry_callback_receives_attempt_info(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []
    client, fake = _make_client_with_fake(
        [_fake_error(openai.InternalServerError), _fake_response("ok")],
        max_retries=3, retry_backoff_seconds=1.0,
        on_retry=lambda attempt, max_retries, wait, exc: calls.append((attempt, max_retries, wait, type(exc))),
    )
    client.call([{"role": "user", "content": "hi"}])
    assert calls == [(0, 3, 1.0, openai.InternalServerError)]


def test_retryable_exceptions_tuple_matches_documented_set():
    assert _RETRYABLE_EXCEPTIONS == (
        openai.InternalServerError, openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
    )
