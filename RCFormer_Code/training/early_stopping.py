"""Validation-NLL early stopping used only after branch selection."""

from __future__ import annotations

import copy


class EarlyStopping:
    def __init__(self, patience):
        self.patience = max(1, int(patience))
        self.best_value = float("inf")
        self.best_epoch = -1
        self.best_state = None
        self.bad_checks = 0

    def update(self, value, model, epoch):
        if float(value) < self.best_value:
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.best_state = copy.deepcopy(
                {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}
            )
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return self.bad_checks >= self.patience

    def restore(self, model):
        if self.best_state is None:
            raise RuntimeError("early stopping has no checkpoint to restore")
        model.load_state_dict(self.best_state)
        return model
