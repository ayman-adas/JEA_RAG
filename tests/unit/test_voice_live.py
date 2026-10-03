from src.voice.gpt_live import ANSWER_TOOL, build_session_config
from src.voice.prompts import build_backend_prompt, build_system_prompt


def test_session_config_uses_live_and_luna(monkeypatch):
    monkeypatch.setenv("GPT_LIVE_MODEL", "gpt-live-1")
    monkeypatch.setenv("GPT_LIVE_VOICE", "cedar")
    monkeypatch.setenv("GPT_LIVE_DELEGATION_MODEL", "gpt-6-luna")

    config = build_session_config(build_system_prompt(), build_backend_prompt())

    assert config["model"] == "gpt-live-1"
    assert config["audio"]["output"]["voice"] == "cedar"
    assert config["audio"]["format"]["rate"] == 24000
    delegation = config["delegation"]["responses"]
    assert delegation["model"] == "gpt-6-luna"
    assert delegation["tools"] == [ANSWER_TOOL]
    assert delegation["tool_choice"] == "auto"
    assert "answer_guild_question" in build_backend_prompt()
    assert "نقابة المهندسين" in build_system_prompt()
