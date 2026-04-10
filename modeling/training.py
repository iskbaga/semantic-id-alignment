import datetime
from pathlib import Path

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, experiment_name, logdir):
        self._experiment_name = experiment_name
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        log_path = self.logdir / (f"{experiment_name}_{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M')}")
        self.writer = SummaryWriter(log_dir=log_path)

    def add_metrics(self, step, metrics):
        logger.info(
            f"step {step}, "
            + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items())
        )
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def close(self):
        self.writer.close()


class EarlyStopper:
    def __init__(self, metric, patience, minimize=True, checkpoints_dir=None, experiment_name=None):
        self.metric = metric
        self.best_metric = None
        self.minimize = minimize
        self.patience = patience
        self.wait = 0
        self.checkpoints_dir = Path(checkpoints_dir)
        self.experiment_name = experiment_name
        self.best_model_file = None
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def check(self, current_metric, model):
        if self.best_metric is None:
            self.best_metric = current_metric
            self.best_model_file = self._save_model(model, current_metric)
            return False

        improved = (self.minimize and current_metric < self.best_metric) or (
            not self.minimize and current_metric > self.best_metric
        )

        if improved:
            if self.best_model_file and self.best_model_file.exists():
                self.best_model_file.unlink()
                logger.info(f"Removed previous best model: {self.best_model_file.name}")

            self.wait = 0
            self.best_metric = current_metric
            self.best_model_file = self._save_model(model, current_metric)
            logger.info(
                f"New best value for {self.metric}: {self.best_metric:.4f} (saved to {self.best_model_file.name})"
            )
            return False
        else:
            self.wait += 1
            logger.info(f"Wait is increased to {self.wait}")

            if self.wait >= self.patience:
                logger.info(
                    f"Patience for {self.metric} is reached: "
                    f"couldn't beat value {self.best_metric:.4f} for {self.wait} calls"
                )
                return True
            return False

    def _save_model(self, model, metric):
        rounded_metric = round(metric, 4)
        filepath = self.checkpoints_dir / f"{self.experiment_name}_best_{rounded_metric}.pth"
        torch.save(model.state_dict(), filepath)
        return filepath

    def get_best_model_path(self):
        return self.best_model_file
