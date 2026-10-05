"""Unit tests for feature_engineering.build_features."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.feature_engineering import (
    HIGH_ENTROPY_DELTA_THRESHOLD,
    WINDOW_SECONDS,
    build_features,
    validate_schema,
)


def _row(ts, run="R-0001", host="H-001", win=0, **kw):
    base = {
        "timestamp": ts.isoformat(),
        "host_id": host,
        "run_id": run,
        "window_id": win,
        "event_type": "process_create",
        "process_name": "chrome.exe",
        "parent_process": "explorer.exe",
        "tree_depth": 2,
        "file_path": None,
        "entropy_before": 4.0,
        "entropy_after": 4.05,
        "entropy_delta": 0.05,
        "registry_key": None,
        "registry_action": None,
        "destination_ip": None,
        "destination_port": None,
        "bytes_sent": 0,
        "bytes_received": 0,
        "auth_result": None,
        "label": 0,
        "scenario": "benign_normal",
        "attack_phase": "BENIGN",
    }
    base.update(kw)
    return base


def test_validate_schema():
    with pytest.raises(ValueError):
        validate_schema(pd.DataFrame([{"foo": 1}]))
    with pytest.raises(ValueError):
        validate_schema(pd.DataFrame(columns=["timestamp", "run_id",
                                              "host_id", "window_id"]))


def test_velocity_and_entropy_thresholds():
    ts = datetime(2026, 9, 29, 10, 0, 0)
    rows = []
    for i in range(60):  # 60 writes in one window -> velocity 2.0
        rows.append(_row(ts + timedelta(seconds=i % 30), event_type="file_write",
                         file_path=f"C:\\a\\{i}.txt",
                         entropy_delta=2.5, entropy_after=6.5))
    df = pd.DataFrame(rows)
    feats = build_features(df)
    assert len(feats) == 1
    assert feats.iloc[0]["file_write_velocity"] == pytest.approx(60 / WINDOW_SECONDS)
    assert feats.iloc[0]["high_entropy_write_count"] == 60
    assert HIGH_ENTROPY_DELTA_THRESHOLD == 1.0


def test_persistence_and_diversity():
    ts = datetime(2026, 9, 29, 10, 0, 0)
    rows = [
        _row(ts, event_type="registry_modify", registry_key="HKCU\\a\\CurrentVersion\\Run",
             registry_action="MODIFY", label=1, scenario="persistence_attack",
             attack_phase="PERSISTENCE"),
        _row(ts + timedelta(seconds=5), event_type="network_connection",
             destination_ip="1.1.1.1", destination_port=443,
             bytes_sent=100, label=1, scenario="persistence_attack",
             attack_phase="PERSISTENCE"),
        _row(ts + timedelta(seconds=6), event_type="authentication",
             auth_result="FAILURE", label=1, scenario="persistence_attack",
             attack_phase="PERSISTENCE"),
    ]
    feats = build_features(pd.DataFrame(rows))
    assert feats.iloc[0]["persistence_key_count"] == 1
    assert feats.iloc[0]["failed_login_count"] == 1
    assert feats.iloc[0]["network_connection_count"] == 1
    assert feats.iloc[0]["signal_diversity"] == 3
    assert feats.iloc[0]["label"] == 1


def test_suspicious_parent_child():
    ts = datetime(2026, 9, 29, 10, 0, 0)
    df = pd.DataFrame([_row(ts, process_name="powershell.exe",
                             parent_process="WINWORD.EXE")])
    feats = build_features(df)
    assert feats.iloc[0]["powershell_count"] == 1
    assert feats.iloc[0]["suspicious_parent_child_count"] == 1
