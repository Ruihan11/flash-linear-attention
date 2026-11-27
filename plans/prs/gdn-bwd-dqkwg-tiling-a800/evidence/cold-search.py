import runpy
import sys
import time
from pathlib import Path

import torch
from autoselect import candidate, chunk_o

mode = sys.argv[1]
fn = candidate() if mode == 'plus' else chunk_o.chunk_bwd_kernel_dqkwg
_, grid, kw = runpy.run_path(str(Path(__file__).with_name('candidate-config.py')))['make'](256)
for k in ('USE_G', 'USE_G_GAMMA', 'USE_DW', 'IS_VARLEN'):
    kw.pop(k)
kw.update(BK=128, BV=128)
torch.cuda.synchronize()
for i in range(2):
    start = time.perf_counter()
    binary = fn[grid](**kw)
    torch.cuda.synchronize()
    print(mode, i, round(time.perf_counter()-start, 5), fn.fn.best_config, len(fn.fn.cache), flush=True)
