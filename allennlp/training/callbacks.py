import logging
import math
import os
import time
import re
import datetime
import traceback
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any, NamedTuple
from enum import Enum

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params, Registrable
from allennlp.common.checks import ConfigurationError
from allennlp.training.moving_average import MovingAverage

class EventType(Enum):
    TRAIN_BEGIN = "train_begin"
    EPOCH_BEGIN = "epoch_begin"
    BATCH_BEGIN = "batch_begin"
    LOSS_BEGIN = "loss_begin"
    BACKWARD_BEGIN = "backward_begin"
    BACKWARD_END = "backward_end"
    STEP_END = "step_end"
    BATCH_END = "batch_end"
    EPOCH_END = "epoch_end"
    TRAIN_END = "train_end"

class Callback(Registrable):
    _priority=0
    def on_event(self, event_type: EventType, data: dict=None) -> None:
        pass

    def set_trainer(self, trainer):
        """Sets the trainer (required)"""
        # TODO: Move to constructor
        self._trainer = _trainer

    @property
    def trainer(self):
        if self._trainer is None:
            raise ValueError("You need to register a trainer with callbacks!!")
        return self._trainer

class CallbackHandler:
    def __init__(self, callbacks=[]):
        self.callbacks = callbacks
        self.callbacks = sorted(self.callbacks, key=lambda o: -getattr(o, '_priority', 0))

    def call(self, attr, *args, **kwargs):
        for cb in self.callbacks: getattr(cb, attr)(*args, **kwargs)

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.call("set_trainer", trainer)

    def on_event(self, event_type: EventType, data: dict=None):
        self.call("on_event", event_type, data=data)

class TensorboardCallback(Callback):
    def __init__(self,
                 serialization_dir: Optional[str] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 log_batch_size_period: Optional[int] = None,
                 moving_average: Optional[MovingAverage] = None) -> None:
        self._batch_num_total = 0
        self._tensorboard = TensorboardWriter(
                get_batch_num_total=lambda: self._batch_num_total,
                serialization_dir=serialization_dir,
                summary_interval=summary_interval,
                histogram_interval=histogram_interval,
                should_log_parameter_statistics=should_log_parameter_statistics,
                should_log_learning_rate=should_log_learning_rate)

    # def on_backward_end(self):
    def on_event(self, event_type: EventType, data: dict={}):
        if self._tensorboard.should_log_histograms_this_batch():
            if event_type == EventType.BACKWARD_END:
                # get the magnitude of parameter updates for logging
                # We need a copy of current parameters to compute magnitude of updates,
                # and copy them to CPU so large models won't go OOM on the GPU.
                # Log these updates after the optimizer step (TODO: Can we do this
                # here?)
                self.param_updates = {name: param.detach().cpu().clone()
                                      for name, param in self.model.named_parameters()}
            elif event_type == EventType.STEP_END:
                param_updates = self.param_updates
                for name, param in self.model.named_parameters():
                    param_updates[name].sub_(param.detach().cpu())
                    update_norm = torch.norm(param_updates[name].view(-1, ))
                    param_norm = torch.norm(param.view(-1, )).cpu()
                    self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                        update_norm / (param_norm + 1e-7))
                self.param_updates = None # release reference

class CheckpointCallback(Callback):
    pass


