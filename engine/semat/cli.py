"""SE Matte command line.

Example (Pro mode, click at x=960,y=400 on frame 0, exclude a point):
  python -m semat.cli pro clip.mp4 --frame 0 --point 960,400 --neg 1700,900 --out renders/

Multiple keyframes: repeat --frame followed by its points, e.g.
  --frame 0 --point 960,400 --frame 200 --point 900,420
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .device import pick_device
from .pipeline_pro import Keyframe, ProJob, run
from .refine import RefineSettings
from .video_io import probe


def _xy(s: str) -> tuple[float, float]:
    x, y = s.split(",")
    return float(x), float(y)


def parse_keyframes(argv: list[str]) -> list[Keyframe]:
    kfs: list[Keyframe] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--frame":
            kfs.append(Keyframe(int(argv[i + 1]), [], []))
            i += 2
            continue
        if a in ("--point", "--neg"):
            if not kfs:
                kfs.append(Keyframe(0, [], []))
            kfs[-1].points.append(_xy(argv[i + 1]))
            kfs[-1].labels.append(1 if a == "--point" else 0)
            i += 2
            continue
        i += 1
    return [k for k in kfs if k.points]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="semat")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="show device info")
    p = sub.add_parser("pro", help="Pro mode: SAM 2.1 + MatAnyone")
    p.add_argument("source")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None, help="exclusive end frame")
    p.add_argument("--quality", default="standard", choices=["standard", "max"])
    p.add_argument("--out", default=None)
    p.add_argument("--outputs", default="prores4444", help="comma list: prores4444,matte,png")
    p.add_argument("--choke", type=float, default=0)
    p.add_argument("--feather", type=float, default=0)
    p.add_argument("--contrast", type=float, default=0)
    p.add_argument("--temporal", type=float, default=0)
    p.add_argument("--no-decontaminate", action="store_true")
    p.add_argument("--device", default=None, help="force cpu/mps/cuda")
    args, rest = ap.parse_known_args(argv)

    dev = pick_device(getattr(args, "device", None))
    if args.cmd == "info":
        print(json.dumps(dev.to_dict(), indent=2))
        return

    info = probe(args.source)
    kfs = parse_keyframes(argv)
    job = ProJob(
        source=args.source, start=args.start, end=args.end or info.nb_frames, keyframes=kfs,
        quality=args.quality, out_dir=args.out, outputs=tuple(args.outputs.split(",")),
        refine=RefineSettings(args.choke, args.feather, args.contrast, args.temporal, not args.no_decontaminate),
    )
    last = [0.0]

    def prog(stage, p, msg):
        now = time.time()
        if now - last[0] > 5.0 or p >= 1.0:
            last[0] = now
            print(f"[{stage:6}] {p * 100:5.1f}%  {msg}", flush=True)

    print(f"Device: {dev.to_dict()}  |  {info.width}x{info.height} @ {float(info.fps):.3f} fps, "
          f"frames {job.start}-{job.end - 1}")
    res = run(job, dev, prog)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
