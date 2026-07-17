"""Firestore order ledger: value codec, order docs, watcher decisions."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ai_video_maker.clients.cloudinary_client import OrderAsset
from ai_video_maker.clients.firebase_client import (
    FirebaseClient,
    decode_fields,
    encode_fields,
    order_from_document,
    resolve_credentials_file,
)
from ai_video_maker.intake import is_order_complete, parse_order_folder

LEAF = "AM-180726-XY12_Dana-Cohen-18.07.2026_10-30"


# ------------------------------ value codec -------------------------------- #

class TestValueCodec:
    def test_decode_typed_values(self):
        fields = {
            "name": {"stringValue": "Dana"},
            "count": {"integerValue": "12"},
            "price": {"doubleValue": 99.5},
            "paid": {"booleanValue": True},
            "extra": {"nullValue": None},
            "nested": {"mapValue": {"fields": {"a": {"stringValue": "b"}}}},
            "tags": {"arrayValue": {"values": [{"stringValue": "x"}]}},
        }
        assert decode_fields(fields) == {
            "name": "Dana", "count": 12, "price": 99.5, "paid": True,
            "extra": None, "nested": {"a": "b"}, "tags": ["x"],
        }

    def test_encode_decode_roundtrip(self):
        data = {"status": "ingested", "project": "dana-cohen",
                "count": 7, "flag": False, "nothing": None}
        assert decode_fields(encode_fields(data)) == data

    def test_bool_is_not_encoded_as_integer(self):
        assert encode_fields({"flag": True})["flag"] == {"booleanValue": True}

    def test_unknown_value_type_decodes_to_none(self):
        assert decode_fields({"geo": {"geoPointValue": {}}}) == {"geo": None}


# ------------------------------- order docs -------------------------------- #

def _doc(fields: dict, doc_id: str = "AM-180726-XY12") -> dict:
    return {
        "name": f"projects/p/databases/(default)/documents/orders/{doc_id}",
        "fields": encode_fields(fields),
        "createTime": "2026-07-18T08:00:00Z",
    }


class TestOrderFromDocument:
    def test_full_doc(self):
        order = order_from_document(_doc({
            "orderId": "AM-180726-XY12", "name": "Dana Cohen",
            "phone": "050-1234567", "email": "dana@example.com",
            "packageId": "premium", "musicMood": "warm piano",
            "blessing": "מזל טוב", "folder": f"video-orders/{LEAF}",
            "status": "new", "createdAt": "2026-07-18T08:00:00Z",
        }))
        assert order.order_id == "AM-180726-XY12"
        assert order.customer == "Dana Cohen"
        assert order.package_id == "premium"
        assert order.status == "new"
        assert order.folder_leaf == LEAF

    def test_minimal_doc_defaults(self):
        order = order_from_document(_doc({}))
        assert order.order_id == "AM-180726-XY12"  # falls back to the doc id
        assert order.customer == ""
        assert order.folder_leaf == ""
        assert order.photo_count is None
        assert order.created_at == "2026-07-18T08:00:00Z"  # createTime fallback

    def test_photo_count_when_present(self):
        order = order_from_document(_doc({"photoCount": 21}))
        assert order.photo_count == 21

    def test_zero_photo_count_treated_as_unknown(self):
        order = order_from_document(_doc({"photoCount": 0}))
        assert order.photo_count is None


# ----------------------------- configuration ------------------------------- #

class _Cfg:
    firebase_credentials_file = ""
    firebase_project_id = ""


class TestCredentialsResolution:
    def test_unconfigured(self, monkeypatch):
        monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.setattr(
            "ai_video_maker.clients.firebase_client.PROJECT_ROOT", Path("/nonexistent")
        )
        assert resolve_credentials_file(_Cfg()) is None
        assert not FirebaseClient.configured(_Cfg())

    def test_env_var_wins_over_default(self, monkeypatch, tmp_path):
        key = tmp_path / "key.json"
        key.write_text(json.dumps({"project_id": "animoment-test"}))
        monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", str(key))
        assert resolve_credentials_file(_Cfg()) == key
        assert FirebaseClient.configured(_Cfg())

    def test_config_path_wins_over_env(self, monkeypatch, tmp_path):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text("{}")
        b.write_text("{}")
        monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", str(b))
        cfg = _Cfg()
        cfg.firebase_credentials_file = str(a)
        assert resolve_credentials_file(cfg) == a

    def test_project_id_read_from_key_file(self, monkeypatch, tmp_path, config):
        key = tmp_path / "key.json"
        key.write_text(json.dumps({"project_id": "animoment-test"}))
        monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", str(key))
        client = FirebaseClient.from_config(config)
        assert client.project_id == "animoment-test"
        assert client.collection == "orders"


# ----------------------- completeness with photo_count ---------------------- #

def _assets(n: int, minutes_ago: float) -> list[OrderAsset]:
    stamp = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    return [
        OrderAsset(public_id=f"x/{i}", url=f"https://x/{i}", format="jpg",
                   position=i, created_at=stamp)
        for i in range(1, n + 1)
    ]


class TestExpectedCountCompleteness:
    def test_exact_count_completes_even_when_fresh(self):
        assert is_order_complete(_assets(3, minutes_ago=0), 10.0, expected_count=3)

    def test_below_count_incomplete_even_when_quiet(self):
        assert not is_order_complete(_assets(2, minutes_ago=60), 10.0, expected_count=3)

    def test_no_expected_count_falls_back_to_quiet_period(self):
        assert is_order_complete(_assets(2, minutes_ago=60), 10.0, expected_count=None)
        assert not is_order_complete(_assets(2, minutes_ago=1), 10.0, expected_count=None)


class TestMoodSuffixFolder:
    def test_mood_suffix_still_parses(self):
        parsed = parse_order_folder(LEAF + "_warm-piano")
        assert parsed["order_id"] == "AM-180726-XY12"
        assert parsed["customer"] == "Dana Cohen"
        assert parsed["stamp"] == "18.07.2026_10-30"
