"""Combine per-tile example shards into one array once every tile is done.

Workers write each tile's examples to its own .npy shard, so no process ever
holds more than one tile of examples. Collation then copies the shards, in
tile order, into a single memory-mapped `examples.npy` and deletes them.
"""

import os

import numpy as np

from .examples import EXAMPLE_DTYPE, N_FEATURES


def collate_shards(shard_paths, out_path, width):
    """Concatenate shards (each (n_i, N_FEATURES, width)) into out_path.

    Returns the total number of examples. Shard paths may be None (tiles
    with no examples) and are skipped.
    """
    shard_paths = [p for p in shard_paths if p]
    sizes = [np.load(p, mmap_mode="r").shape[0] for p in shard_paths]
    total = sum(sizes)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=EXAMPLE_DTYPE,
                                    shape=(total, N_FEATURES, width))
    row = 0
    for path, n in zip(shard_paths, sizes):
        out[row:row + n] = np.load(path)
        row += n
        os.remove(path)
    out.flush()
    del out
    return total
