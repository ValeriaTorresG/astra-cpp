"""logging, stage timing, and resource monitoring (rss, cpu, node memory)."""
from __future__ import annotations

import contextlib
import logging
import os
import resource
import sys
import threading
import time
from pathlib import Path


def gib(nbytes: float) -> str:
    return f'{nbytes / 2**30:.2f} GiB'


def peak_rss_bytes() -> int:
    # on linux, ru_maxrss is in kib
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _proc_status() -> dict:
    out = {}
    try:
        with open('/proc/self/status', encoding='ascii') as fh:
            for line in fh:
                if line.startswith(('VmRSS:', 'Threads:')):
                    key, value = line.split(':', 1)
                    out[key] = int(value.split()[0]) * (1 if key == 'Threads' else 1024)
    except OSError:
        pass
    return out


def mem_available() -> int | None:
    try:
        with open('/proc/meminfo', encoding='ascii') as fh:
            for line in fh:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def setup_logger(log_path: Path | None = None, name: str = 'astra_cpp') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    for handler in handlers:
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


class ResourceMonitor:
    """record stage times and resource usage every `interval` s in a separate thread.

    c++ stages release the gil, so the heartbeat continues while cgal
    triangulates.
    """

    def __init__(self, logger: logging.Logger, interval: float = 30.0):
        self.logger = logger
        self.interval = interval
        self.timings: dict[str, float] = {}
        self.stage_name = 'init'
        self._t0 = time.monotonic()
        self._last = (self._t0, time.process_time())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self, label: str | None = None) -> None:
        status = _proc_status()
        wall, cpu = time.monotonic(), time.process_time()
        dt = wall - self._last[0]
        cores = (cpu - self._last[1]) / dt if dt > 0 else 0.0
        self._last = (wall, cpu)
        parts = [f'stage={label or self.stage_name}', f'elapsed={wall - self._t0:.1f}s',
                 f'rss={gib(status.get("VmRSS", 0))}', f'peak_rss={gib(peak_rss_bytes())}',
                 f'cpu_cores={cores:.1f}', f'threads={status.get("Threads", 0)}']
        available = mem_available()
        if available is not None:
            parts.append(f'node_mem_available={gib(available)}')
        self.logger.info('resources %s', ' '.join(parts))

    @contextlib.contextmanager
    def stage(self, name: str):
        self.stage_name = name
        self.logger.info('[%s] inicio', name)
        t0 = time.perf_counter()
        yield
        elapsed = time.perf_counter() - t0
        self.timings[name] = self.timings.get(name, 0.0) + elapsed
        self.logger.info('[%s] %.2f s | rss=%s peak_rss=%s', name, elapsed,
                         gib(_proc_status().get('VmRSS', 0)), gib(peak_rss_bytes()))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.sample()

    def __enter__(self) -> 'ResourceMonitor':
        self.sample('start')
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self.sample('finish')


def host_info() -> dict:
    return {
        'host': os.uname().nodename,
        'pid': os.getpid(),
        'affinity_cores': len(os.sched_getaffinity(0)),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
    }
