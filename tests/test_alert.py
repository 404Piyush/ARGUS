"""Alert tests: evidence parity + dtype fix + selection."""
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.argus_alert import collect_evidence, select_suspicious


def test_evidence_parity_with_risk():
    # Every risk rule should have a corresponding evidence string.
    row = {
        "powershell_count": 1, "cmd_count": 1,
        "suspicious_parent_child_count": 1, "file_write_velocity": 2.0,
        "high_entropy_write_count": 10, "entropy_delta_mean": 2.0,
        "persistence_key_count": 1, "network_connection_count": 10,
        "failed_login_count": 5,
    }
    ev = collect_evidence(row)
    assert len(ev) == 9, f"expected 9 evidence items, got {ev}"
    assert any("High-entropy" in e for e in ev)


def test_evidence_empty():
    assert collect_evidence({}) == ["No strong behavioral indicators"]


def test_select_suspicious():
    df = pd.read_csv(BASE_DIR / "data" / "processed" / "argus_features.csv")
    idx = select_suspicious(df, scenario="ransomware_like", top_n=1)
    assert len(idx) == 1
    assert df.loc[idx[0], "scenario"] == "ransomware_like"


def test_alert_main_runs():
    from src.argus_alert import main
    out = main(scenario="ransomware_like", top_n=1)
    assert out["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert 0 <= out["probability"] <= 1
