import asyncio

from kontexta_cli.core import session as session_mod


class Thought:
    def __init__(self, text):
        self.text = text


class Text:
    def __init__(self, text):
        self.text = text


class ToolCall:
    text = "ignored"


class FakeStatus:
    def __init__(self):
        self.updates = []
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True

    def update(self, msg):
        self.updates.append(msg)


class FakeConsole:
    width = 60

    def __init__(self):
        self.status_obj = FakeStatus()

    def status(self, *a, **k):
        return self.status_obj


class TypedStream:
    def __init__(self, items):
        self._items = items

    @property
    def chunks(self):
        async def gen():
            for i in self._items:
                yield i
        return gen()


def _consume(session, stream, monkeypatch):
    fake = FakeConsole()
    monkeypatch.setattr(session_mod, "console", fake)
    monkeypatch.setattr(session_mod, "RICH_AVAILABLE", True)
    session.json_output = False
    session.render_markdown = False
    result = asyncio.run(session._consume_token_stream(stream))
    return fake.status_obj, result


def test_thoughts_update_one_spinner_line_and_stay_out_of_response(session, monkeypatch):
    stream = TypedStream([Thought("let me [think]\n"), Thought("about   this"), ToolCall(), Text("Hello"), Text(" world")])
    status, (text, count, t_first) = _consume(session, stream, monkeypatch)
    assert text == "Hello world" and count == 2 and t_first is not None
    assert len(status.updates) == 2
    assert "\n" not in status.updates[-1]
    assert "thinking:" in status.updates[-1] and "about this" in status.updates[-1]
    assert "\\[think]" in status.updates[-1]
    assert status.stopped


def test_long_thought_is_cut_to_terminal_width(session, monkeypatch):
    stream = TypedStream([Thought("x" * 500), Text("ok")])
    status, _ = _consume(session, stream, monkeypatch)
    line = status.updates[-1]
    assert line.count("x") <= FakeConsole.width - 16


def test_plain_string_stream_still_works(session, monkeypatch):
    async def gen():
        for t in ("a", "b"):
            yield t

    status, (text, count, _) = _consume(session, gen(), monkeypatch)
    assert text == "ab" and count == 2 and status.updates == []
