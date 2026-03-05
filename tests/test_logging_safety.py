from __future__ import annotations

from types import SimpleNamespace

from utils.logging import format_text_record


def test_format_text_record_escapes_braces_and_does_not_raise() -> None:
    record = {
        "level": SimpleNamespace(name="WARNING"),
        "module": "fetcher",
        "function": "get_orderbook",
        "line": 514,
        "message": 'Orderbook failed: 401-{"code":"400003","msg":"KC-API-KEY not exists"}',
    }

    template = format_text_record(record)
    rendered = template.format_map({})
    assert '{"code":"400003","msg":"KC-API-KEY not exists"}' in rendered


def test_format_text_record_handles_missing_fields() -> None:
    record = {"message": "Regime payload: {'trend': 'bull'}"}
    template = format_text_record(record)
    rendered = template.format_map({})
    assert "Regime payload" in rendered

