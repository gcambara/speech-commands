from bisect import bisect_right
from torch.optim.lr_scheduler import _LRScheduler

class ConsecutiveLR(_LRScheduler):
    """Similar to PyTorch's Sequential LR, but every newly activated scheduler
       gets the initial LR at the last value from the previous scheduler.
    """

    def __init__(self, optimizer, schedulers, milestones, last_epoch=-1, verbose=False):
        for scheduler_idx in range(len(schedulers)):
            if schedulers[scheduler_idx].optimizer != optimizer:
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    f"got schedulers at index {scheduler_idx} to be different than the optimizer passed in."
                )

            if (schedulers[scheduler_idx].optimizer != schedulers[0].optimizer):
                raise ValueError(
                    "Sequential Schedulers expects all schedulers to belong to the same optimizer, but "
                    f"got schedulers at index {0} and {scheduler_idx} to be different."
                )
        if (len(milestones) != len(schedulers) - 1):
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                "than the number of milestone points, but got number of schedulers {} and the "
                "number of milestones to be equal to {}".format(len(schedulers), len(milestones))
            )
        self._schedulers = schedulers
        self._milestones = milestones
        self.last_epoch = last_epoch + 1
        self.optimizer = optimizer
        self._last_lr = schedulers[0].get_last_lr()
        self._current_idx = 0

    def step(self):
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)

        if idx != self._current_idx:
            self._schedulers[idx].base_lrs = self._schedulers[idx - 1].get_last_lr()
            self._current_idx = idx
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            self._schedulers[idx].step(0)
        else:
            self._schedulers[idx].step()
        self._last_lr = self._schedulers[idx].get_last_lr()

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_schedulers')}
        state_dict['_schedulers'] = [None] * len(self._schedulers)

        for idx, s in enumerate(self._schedulers):
            state_dict['_schedulers'][idx] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        _schedulers = state_dict.pop('_schedulers')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['_schedulers'] = _schedulers

        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)
