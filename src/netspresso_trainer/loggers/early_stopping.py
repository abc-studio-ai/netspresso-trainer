import torch
from typing import  Callable
from ..losses.builder import LossFactory
from ..metrics.builder import MetricFactory
import operator

class EarlyStopping:
    mode_dict = {"min": operator.lt, "max": operator.gt}
    order_dict = {"min": "<", "max": ">"}
    def __init__(
            self,
            monitor: str,
            mode: str ,
            min_delta=0.00, 
            patience=3,
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.wait_count = 0
        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        self.best_score = float('inf') if self.monitor_op == operator.lt else float('-inf')
    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]
    def _validate_monitor_metric(self, metric_names):
        error_msg = (f"Early stopping error: {self.monitor} is not available. It should be one of {','.join(metric_names)}")
        if self.monitor != "loss" and self.monitor not in metric_names :
            raise Exception(error_msg)
        
    def _improvement_message(self, current) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if self.best_score != float('inf') and self.best_score != float('-inf'):
            msg = (
            f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
            f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg
    def early_stopping_check(self, loss_factory: LossFactory , metrics_factory:MetricFactory):
        should_stop = None
        reason= None

        if self.monitor == "loss":
            current = loss_factory.result('valid').get('total')
        else:
            self._validate_monitor_metric(metrics_factory.metric_names)
            valid_metrics = metrics_factory.result('valid')
            current = valid_metrics[self.monitor]['mean']
        print("current: ",current, "best: ", self.best_score)
        if self.monitor_op(current - self.min_delta, self.best_score):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                )
        return should_stop, reason
    
def build_early_stopping(conf):
    if conf.logging.early_stopping:
        early_stopping = EarlyStopping(
            conf.logging.early_stopping.monitor,
            conf.logging.early_stopping.mode,
            conf.logging.early_stopping.min_delta,
            conf.logging.early_stopping.patience
        )
        return early_stopping
    else:
        return None
