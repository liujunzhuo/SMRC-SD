# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate only live trajectories and restore their environment indices."""

import numpy as np

from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions


def generate_balanced_active_batch(batch, worker_group):
    """Balance padded live requests by prompt length, then undo both operations."""
    padded, pad_size = pad_dataproto_to_divisor(batch, worker_group.world_size)
    lengths = padded.batch["attention_mask"].sum(-1).tolist()
    partitions = get_seqlen_balanced_partitions(lengths, worker_group.world_size, equal_size=True)
    order = np.array([i for part in partitions for i in part], dtype=np.int64)
    output = worker_group.generate_sequences(padded.select_idxs(order))
    output = output.select_idxs(np.argsort(order))
    return unpad_dataproto(output, pad_size=pad_size), pad_size


def restore_environment_batch(batch, active_indices, full_size):
    """Inactive placeholders are discarded by active_masks before training."""
    active_indices = np.asarray(active_indices, dtype=np.int64)
    if not len(active_indices):
        raise ValueError("Cannot restore an empty live batch")
    inverse = np.zeros(full_size, dtype=np.int64)
    inverse[active_indices] = np.arange(len(active_indices))
    return batch.select_idxs(inverse)
