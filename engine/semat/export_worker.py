"""Runs the export stage (look controls + ProRes/PNG writing) in its own process, so its CPU work
doesn't compete for Python's GIL with the edge stage. It picks up refined frames as they appear
(they are written atomically).

argv[1] = pickle with job/info/alpha_dir. stdout: "X <frame>" per frame, "RES <json>" at the end,
"ERR <msg>" on failure.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import traceback


class _Board:
    def tick(self, stage):
        sys.stdout.write("X\n")
        sys.stdout.flush()


class _Wait:
    def __init__(self, d, parent):
        self.d, self.parent = d, parent

    def wait(self, i):
        p = os.path.join(self.d, f"{i:06d}.png")
        while not os.path.exists(p):
            if os.getppid() != self.parent:  # the engine went away
                sys.exit(1)
            time.sleep(0.02)


def main():
    import signal
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # run cleanup (remove half-written files)
    try:
        with open(sys.argv[1], "rb") as f:
            a = pickle.load(f)
        from .pipeline_pro import _stage_export

        res = _stage_export(a["job"], a["info"], a["alpha_dir"], _Board(), _Wait(str(a["alpha_dir"]), a["parent"]))
        sys.stdout.write("RES " + json.dumps(res) + "\n")
        sys.stdout.flush()
    except BaseException as e:  # noqa: BLE001
        traceback.print_exc()
        sys.stdout.write("ERR " + f"{type(e).__name__}: {e}".replace("\n", " ") + "\n")
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
