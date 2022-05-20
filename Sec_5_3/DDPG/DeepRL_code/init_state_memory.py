import numpy as np
import torch
from torch import nn

class InitStateBuffer:

    def __init__(self, buffer_size, state_shape, device):
        self._p = 0
        self._n = 0
        self.buffer_size = buffer_size

        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state):
        self.states[self._p].copy_(torch.from_numpy(state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes]
        )