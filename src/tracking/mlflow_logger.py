from __future__ import annotations

from typing import Any

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


class MLFlowLogger:
    """Minimal MLflow logger used for experiment tracking."""

    def __init__(self, experiment_name: str = "warehouse-marl") -> None:
        self.experiment_name = experiment_name
        self._active_run = None
        if mlflow is not None:
            mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> Any:
        if mlflow is None:
            self._active_run = None
            return None
        self._active_run = mlflow.start_run(run_name=run_name)
        return self._active_run

    def log_params(self, params: dict[str, Any]) -> None:
        if mlflow is not None and self._active_run is not None:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if mlflow is not None and self._active_run is not None:
            mlflow.log_metrics(metrics)

    def end_run(self) -> None:
        if mlflow is not None and self._active_run is not None:
            mlflow.end_run()
            self._active_run = None
