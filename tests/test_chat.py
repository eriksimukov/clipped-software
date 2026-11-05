import json

from clipped.pipeline.chat import parse_chat_file, parse_twitch_rechat, parse_youtube_live_chat


def test_twitch_rechat_reads_offset_and_body():
    text = json.dumps(
        {
            "comments": [
                {"content_offset_seconds": 12.5, "message": {"body": "KEKW"}},
                {"message": {"body": "ohne Zeitstempel"}},
            ]
        }
    )
    assert parse_twitch_rechat(text) == [{"t": 12.5, "text": "KEKW"}]


def test_youtube_live_chat_skips_broken_lines_and_non_text():
    good = {
        "replayChatItemAction": {
            "videoOffsetTimeMsec": "2000",
            "actions": [
                {
                    "addChatItemAction": {
                        "item": {
                            "liveChatTextMessageRenderer": {
                                "message": {"runs": [{"text": "no "}, {"text": "way"}]}
                            }
                        }
                    }
                },
                {"addChatItemAction": {"item": {"liveChatPaidMessageRenderer": {}}}},
            ],
        }
    }
    text = "\n".join([json.dumps(good), "kein json", json.dumps({"other": 1})])
    assert parse_youtube_live_chat(text) == [{"t": 2.0, "text": "no way"}]


def test_parse_chat_file_picks_parser_by_name_and_sorts(tmp_path):
    path = tmp_path / "source.rechat.json"
    path.write_text(
        json.dumps(
            {
                "comments": [
                    {"content_offset_seconds": 5, "message": {"body": "b"}},
                    {"content_offset_seconds": 1, "message": {"body": "a"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert [m["text"] for m in parse_chat_file(path)] == ["a", "b"]
