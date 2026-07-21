"""
cli_common.make_transcript_writer: the only thing worth unit-testing
here is that (a) tool-call arguments and tool-result content come back
as real nested JSON, not escaped strings — that's the entire point of
this writer over just dumping the raw messages list — and (b) repeated
calls for the same agent_name number sequentially instead of clobbering
each other, since a single run can invoke e.g. "modeling" more than once.
"""
import json

from agentic_ml.cli_common import make_transcript_writer


def _sample_messages():
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Do the thing."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "1", "type": "function",
                 "function": {"name": "get_facts", "arguments": '{"x": 1}'}},
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": '{"fact": "value", "n": 2}'},
        {"role": "assistant", "content": '{"verdict": "approved"}'},
    ]


def test_transcript_writer_parses_tool_call_arguments_as_real_json(tmp_path):
    write = make_transcript_writer(tmp_path)
    path = write("modeling", _sample_messages())
    transcript = json.loads(path.read_text())

    assistant_with_tool_call = transcript[2]
    args = assistant_with_tool_call["tool_calls"][0]["function"]["arguments"]
    assert args == {"x": 1}  # a real dict, not the string '{"x": 1}'


def test_transcript_writer_parses_tool_result_content_as_real_json(tmp_path):
    write = make_transcript_writer(tmp_path)
    path = write("modeling", _sample_messages())
    transcript = json.loads(path.read_text())

    tool_message = transcript[3]
    assert tool_message["content"] == {"fact": "value", "n": 2}


def test_transcript_writer_parses_final_assistant_json_content(tmp_path):
    write = make_transcript_writer(tmp_path)
    path = write("modeling", _sample_messages())
    transcript = json.loads(path.read_text())

    final_message = transcript[4]
    assert final_message["content"] == {"verdict": "approved"}


def test_transcript_writer_leaves_plain_text_content_alone(tmp_path):
    write = make_transcript_writer(tmp_path)
    path = write("modeling", _sample_messages())
    transcript = json.loads(path.read_text())

    assert transcript[0]["content"] == "You are an agent."
    assert transcript[1]["content"] == "Do the thing."


def test_transcript_writer_numbers_files_per_agent_name(tmp_path):
    write = make_transcript_writer(tmp_path)
    path1 = write("modeling", _sample_messages())
    path2 = write("modeling", _sample_messages())
    path3 = write("verification", _sample_messages())

    assert path1.name == "modeling_01.json"
    assert path2.name == "modeling_02.json"
    assert path3.name == "verification_01.json"
    assert path1 != path2
    assert path1.exists() and path2.exists() and path3.exists()


def test_transcript_writer_creates_transcripts_subdir(tmp_path):
    write = make_transcript_writer(tmp_path)
    path = write("intake", _sample_messages())
    assert path.parent == tmp_path / "transcripts"
