import datetime
from pathlib import Path

from loguru import logger
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, experiment_name, logdir):
        self._experiment_name = experiment_name
        self._timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)

        log_path = self.logdir / f"{self._experiment_name}_{self._timestamp}"
        self.writer = SummaryWriter(log_dir=log_path)

    def add_metrics(self, step, metrics):
        logger.info(
            f"step {step}, "
            + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items())
        )
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)

    def get_timestamp(self):
        return self._timestamp

    def close(self):
        self.writer.close()
