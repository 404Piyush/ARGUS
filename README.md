# ARGUS

Synthetic endpoint-detection pipeline: **simulator → features → XGBoost → risk engine → SOC alert**, plus a real-data baseline.

## Quickstart

```bash
pip install -r requirements.txt
make data features train alert
# fast tests (<30s)
make test-fast
# full regen e2e (74k events, 2880 windows, ~2-5 min)
make test-e2e
```

Run order:
1. `python3 simulator/generator.py` — 8 scenarios × 60 runs × 6 windows (30s) → `data/raw/synthetic/*.jsonl` + `all_synthetic_telemetry.jsonl` (74,524 events). Use `make data`.
2. `python3 src/feature_engineering.py` — windowed 30-feature matrix → `data/processed/argus_features.csv` (2880 rows). Use `make features`.
3. `python3 src/train_model.py [--no-plots] [--test-size 0.25] [--seed 42]` — GroupShuffleSplit by `run_id`, XGBoost (250, depth 5) → `models/argus_xgboost.json`, `results/metrics.json`, PNGs. Use `make train`.
4. `python3 -m src.argus_alert [--scenario ransomware_like] [--top-n 1]` — demo SOC alert. Use `make alert`.
5. `python3 argus_baseline.py` — side-branch on real CIC-MalMem-2022 (`MalMem2022.csv` from HuggingFace `bvk/CIC-MalMem-2022` / Kaggle / UNB) → root `confusion_matrix.png`, `roc_curve.png`, `shap_summary.png`.

## Structure

- `simulator/generator.py` — scenarios: `benign_normal`, `benign_high_entropy`, `powershell_attack`, `persistence_attack`, `ransomware_like`, `multi_stage_attack` (kill-chain 0-1 benign, 2 exec, 3 persist, 4 C2, 5 impact), `brute_force_attack` (4-6 FAILURE + SUCCESS), `c2_heavy_attack` (8-12 beacons). Fixed: multi-stage windows now keep `scenario=multi_stage_attack` (was leaking to `ransomware_like`/`benign_*`); persistence regex fixed (`CurrentVersion\Run` single-backslash).
- `src/feature_engineering.py` — 30 features (`WINDOW_SECONDS=30`, `HIGH_ENTROPY_DELTA_THRESHOLD=1.0`, `validate_schema()`). See `src/` for full list.
- `src/train_model.py` — `train_and_evaluate()` testable core, `metrics.json` for CI gates. Current: P 0.998 / R 1.0 / F1 0.999 / AUC 1.0.
- `src/risk_engine.py` — `PROB_WEIGHT=60` + `WEIGHTS` table + `THRESHOLDS` (75/50/25), `explain_risk()`.
- `src/argus_alert.py` — `collect_evidence()` in parity with risk weights (incl. high-entropy writes), `select_suspicious()`, robust imports (`python -m src.argus_alert` from root or `python src/argus_alert.py`).
- `data/`, `models/argus_xgboost.json`, `results/` (argus_* PNGs + `feature_importance.csv` + `metrics.json`).

## Tests

- `pytest -m "not e2e"` — 26 unit/integration (schema, velocity, persistence, split, risk table, alert).
- `pytest -m e2e` — full `generator.main()` regen + leakage check + train gate (≥0.90, within 0.05 of `tests/golden/baseline_metrics.json`).
- `pytest` — all 29.

## Known gaps / next

- Model still near-perfect (1.0 AUC); 11 features (network/auth/registry) at 0.0 importance — need harder negatives where those alone distinguish.
- No hyperparameter search; thresholds uncalibrated; `argus_baseline.py` still standalone (no shared utils).
