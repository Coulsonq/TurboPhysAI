# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
# The index mapping is adapted from a local ACT-H20 experiment whose ownership
# and license remain to be confirmed before distribution.

"""Vectorized episode-to-frame mapping with unchanged shuffle and resume order."""

import functools

import numpy as np
import torch


def sampler_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, epoch, start):
        if not isinstance(start, int) or start < 0:
            yield from original(self, epoch, start)
            return

        if self.shuffle:
            order = torch.randperm(self._num_frames, generator=self._epoch_generator(epoch))
            positions = order.numpy()[start:]
        else:
            positions = np.arange(start, self._num_frames, dtype=np.int64)

        episodes = np.searchsorted(self._cum_lengths, positions, side="right")
        previous = np.concatenate((np.zeros(1, dtype=self._cum_lengths.dtype), self._cum_lengths[:-1]))
        indices = (self._starts[episodes] + positions - previous[episodes]).tolist()
        mapping = self._absolute_to_relative
        if mapping is not None:
            if any(index not in mapping for index in indices):
                yield from original(self, epoch, start)
                return
            indices = [mapping[index] for index in indices]
        yield from indices

    return wrapped
