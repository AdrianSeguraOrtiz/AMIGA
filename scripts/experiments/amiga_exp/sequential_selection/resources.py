"""CPU budgets and isolated worker affinity for parallel training jobs."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys


def cpu_limits():
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    quota = None
    # Respect any v2 quota in this process's cgroup or its ancestors.
    membership = Path('/proc/self/cgroup')
    root = Path('/sys/fs/cgroup')
    if membership.exists():
        for line in membership.read_text().splitlines():
            if line.startswith('0::'):
                leaf = root / line.split('::', 1)[1].lstrip('/')
                for folder in [leaf, *leaf.parents]:
                    if folder != root and root not in folder.parents:
                        continue
                    path = folder/'cpu.max'
                    if path.exists():
                        value, period = path.read_text().split()[:2]
                        if value != 'max':
                            bound = int(value)/int(period)
                            quota = bound if quota is None else min(quota, bound)
    capacity = min(len(cpus), max(1, math.floor(quota))) if quota is not None else len(cpus)
    return dict(allowed_cpus=cpus, cpu_quota=quota, capacity=capacity)


def layout(jobs, threads, limits=None):
    limits = cpu_limits() if limits is None else limits
    if not isinstance(jobs, int) or not isinstance(threads, int) or jobs < 1 or not 1 <= threads <= 8:
        raise ValueError('Use positive workers and 1–8 threads per worker')
    if jobs*threads > limits['capacity']:
        raise ValueError('Worker/thread product exceeds available CPU capacity')
    cpus = limits['allowed_cpus']
    return dict(workers=jobs, threads=threads, cpu_budget=jobs*threads,
                available_cpu_capacity=limits['capacity'], cpu_quota=limits['cpu_quota'],
                affinity_supported=hasattr(os, 'sched_setaffinity'),
                cpu_sets=[cpus[i*threads:(i+1)*threads] for i in range(jobs)])


def worker_environment(threads):
    return dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads),
                MPLBACKEND='Agg')


def worker_command(module, arguments, cpus):
    if not cpus or any(not isinstance(c, int) or c < 0 for c in cpus):
        raise ValueError('Invalid CPU affinity')
    # Set affinity before importing NumPy or a native training library. Native
    # thread pools inherit this worker's CPU set and cannot invade other slots.
    bootstrap = (f'import os,runpy; '
                 f'os.sched_setaffinity(0,{set(cpus)!r}) if hasattr(os,"sched_setaffinity") else None; '
                 f'runpy.run_module({module!r},run_name="__main__")')
    return [sys.executable, '-c', bootstrap, *map(str, arguments)]
