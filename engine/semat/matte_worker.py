"""Runs the MatAnyone stage in its own process.

Python's GIL lets the edge/export threads starve the GPU thread (soft matte dropped from 2.1 to
1.2 fps when sharing a process); in a separate process it keeps ~1.8 fps while the Neural Engine
refines edges in parallel.

Protocol: argv[1] = pickle file with the stage arguments. stdout lines:
  M <frame>   frame written      ALL   everything was cached      DONE   finished
  ERR <msg>   failed
"""
from __future__ import annotations

import pickle
import sys
import traceback


class _Out:
    def mark(self, i):
        sys.stdout.write(f"M {i}\n")
        sys.stdout.flush()

    def mark_all(self, idxs):
        sys.stdout.write("ALL\n")
        sys.stdout.flush()


class _Board:
    def tick(self, stage):
        pass

    def skip(self, stage):
        pass


def main():
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # run cleanup (remove half-written files)
    try:
        with open(sys.argv[1], "rb") as f:
            a = pickle.load(f)
        from .device import pick_device
        from .pipeline_pro import _stage_matte

        dev = pick_device(a["device"])
        _stage_matte(a["job"], a["info"], dev, a["kfs"], a["masks"], a["size"], a["dir"], _Board(), _Out())
        sys.stdout.write("DONE\n")
        sys.stdout.flush()
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        msg = f"{type(e).__name__}: {e}".replace("\n", " ")
        sys.stdout.write(f"ERR {msg}\n")
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
