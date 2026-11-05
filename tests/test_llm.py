import json

import httpx
import pytest

from clipped.config import Llm
from clipped.pipeline import llm

CANDIDATE = {"start": 0.0, "end": 30.0, "peak": 10.0, "text": "hello"}
VALID = {
    "hook": 7,
    "payoff": 6,
    "emotion": 5,
    "standalone": 4,
    "gta6_relevance": 9,
    "reason": "r",
    "hooks": ["a", "b", "c"],
    "title": "t",
    "description": "d",
    "hashtags": ["#x"],
}


def _response(status: int, content: str = "") -> httpx.Response:
    body = {"message": {"content": content}} if status == 200 else {"error": "CUDA error"}
    return httpx.Response(status, text=json.dumps(body), request=httpx.Request("POST", "http://x"))


@pytest.fixture
def replies(monkeypatch):
    """Spielt die angegebenen Antworten der Reihe nach ab und merkt sich die Aufrufe."""
    queue: list = []
    calls: list = []

    def fake_post(url, json, timeout):
        calls.append(json)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    monkeypatch.setattr(llm, "OLLAMA_RETRY_SECONDS", 0)
    return queue, calls


def test_server_error_is_retried(replies):
    queue, calls = replies
    queue += [_response(500), _response(200, json.dumps(VALID))]
    assert llm.assess(CANDIDATE, "S", Llm()).hook == 7
    assert len(calls) == 2


def test_persistent_server_error_is_readable(replies):
    queue, calls = replies
    queue += [_response(500) for _ in range(llm.OLLAMA_ATTEMPTS)]
    with pytest.raises(RuntimeError, match="Ollama meldet Fehler 500.*CUDA error"):
        llm.assess(CANDIDATE, "S", Llm())
    assert len(calls) == llm.OLLAMA_ATTEMPTS


def test_client_error_is_not_retried(replies):
    queue, calls = replies
    queue.append(_response(404))
    with pytest.raises(RuntimeError, match="404"):
        llm.assess(CANDIDATE, "S", Llm())
    assert len(calls) == 1


def test_connection_problem_is_readable(replies):
    queue, _ = replies
    queue.append(httpx.ConnectError("verweigert"))
    with pytest.raises(RuntimeError, match="nicht erreichbar"):
        llm.assess(CANDIDATE, "S", Llm())


def test_invalid_answer_is_retried_then_reported_as_candidate_problem(replies):
    queue, calls = replies
    queue += [
        _response(200, "kein json"),
        _response(200, json.dumps({"hook": 99})),
        _response(200, "{}"),
    ]
    # ValueError: nur dieser Kandidat scheitert, der Lauf geht weiter.
    with pytest.raises(ValueError, match="Keine gültige Antwort"):
        llm.assess(CANDIDATE, "S", Llm())
    assert len(calls) == 3


def test_request_uses_schema_and_disables_thinking(replies):
    queue, calls = replies
    queue.append(_response(200, json.dumps(VALID)))
    llm.assess(CANDIDATE, "Streamer", Llm(model="m"))
    payload = calls[0]
    assert payload["model"] == "m"
    assert payload["think"] is False
    assert payload["format"]["properties"]["hook"]["maximum"] == 10
    assert "Streamer" in payload["messages"][1]["content"]
    assert "hello" in payload["messages"][1]["content"]


def test_context_tells_the_model_whether_the_section_is_confirmed(replies):
    queue, calls = replies
    queue += [_response(200, json.dumps(VALID)), _response(200, json.dumps(VALID))]
    llm.assess(CANDIDATE, "S", Llm(), section_confirmed=True)
    llm.assess(CANDIDATE, "S", Llm())
    assert "This part of the stream shows GTA 6." in calls[0]["messages"][1]["content"]
    assert "may be about something else" in calls[1]["messages"][1]["content"]
