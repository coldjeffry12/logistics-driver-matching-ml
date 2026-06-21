"""Checks for deployment-time artifact readiness."""

from src import runtime_setup


def test_runtime_setup_detects_ready_artifacts(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "model.joblib"
    drivers_path = tmp_path / "drivers.csv"
    monkeypatch.setattr(runtime_setup, "MODEL_PATH", model_path)
    monkeypatch.setattr(runtime_setup, "DRIVERS_PATH", drivers_path)

    assert not runtime_setup.runtime_artifacts_ready()

    model_path.touch()
    drivers_path.touch()

    assert runtime_setup.runtime_artifacts_ready()
    assert runtime_setup.ensure_runtime_artifacts() is False


def test_runtime_setup_creates_missing_artifacts(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "models" / "model.joblib"
    drivers_path = tmp_path / "data" / "drivers.csv"
    shipments_path = tmp_path / "data" / "shipments.csv"
    matches_path = tmp_path / "data" / "matches.csv"
    processed_path = tmp_path / "data" / "processed.csv"
    database_path = tmp_path / "data" / "logistics.db"

    for name, path in {
        "MODEL_PATH": model_path,
        "DRIVERS_PATH": drivers_path,
        "SHIPMENTS_PATH": shipments_path,
        "MATCHES_PATH": matches_path,
        "PROCESSED_PATH": processed_path,
        "DATABASE_PATH": database_path,
    }.items():
        monkeypatch.setattr(runtime_setup, name, path)

    def fake_generate_data() -> None:
        for path in (
            drivers_path,
            shipments_path,
            matches_path,
            processed_path,
            database_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def fake_train_model() -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()

    monkeypatch.setattr(runtime_setup, "_generate_data", fake_generate_data)
    monkeypatch.setattr(runtime_setup, "_train_model", fake_train_model)

    assert runtime_setup.ensure_runtime_artifacts() is True
    assert runtime_setup.runtime_artifacts_ready()
