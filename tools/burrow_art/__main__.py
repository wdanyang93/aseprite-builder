"""Command line entry point: `python3 -m burrow_art ...`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .measure import measure
from .normalize import normalize
from .spec import Spec


def _load_spec(request: str | None) -> Spec:
    return Spec.from_request(request) if request else Spec()


def _print_report(report, title: str) -> bool:
    print(f"\n## {title}")
    print(report.as_markdown())
    print(f"\n판정: {'PASS' if report.passed else 'FAIL'}")
    return report.passed


def _cmd_verify(args: argparse.Namespace) -> int:
    spec = _load_spec(args.request)
    ok = True
    for path in args.frames:
        ok &= _print_report(measure(path, spec), Path(path).name)
    return 0 if ok else 1


def _cmd_normalize(args: argparse.Namespace) -> int:
    spec = _load_spec(args.request)
    out_dir = Path(args.out_dir)
    ok = True

    for path in args.frames:
        source = Path(path)
        destination = out_dir / source.name
        result = normalize(
            source,
            destination,
            spec,
            margin=args.margin,
            alpha_threshold=args.alpha_threshold,
        )

        print(f"\n## {source.name} → {destination}")
        print(f"- 원본 피사체: {result.source_size[0]} × {result.source_size[1]}")
        print(
            f"- 배치: {result.placed_size[0]} × {result.placed_size[1]} "
            f"(배율 {result.scale:.4f}) / 캔버스 {spec.width} × {spec.height}"
        )
        print(f"- 알파 이진화: 부분 알파 {result.binarised_px:,}px 해소")
        print(f"- 빨강 프린지 복구: {result.red_repaired_px:,}px")
        for note in result.notes:
            print(f"- {note}")

        ok &= _print_report(measure(destination, spec), f"{destination.name} 검수")

    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="burrow_art", description=__doc__)
    parser.add_argument(
        "-r", "--request", help="REQUEST.json to read the verify rules from"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="measure frames against the spec")
    verify.add_argument("frames", nargs="+")
    verify.set_defaults(func=_cmd_verify)

    fix = sub.add_parser("normalize", help="rewrite frames to satisfy the spec")
    fix.add_argument("frames", nargs="+")
    fix.add_argument("-o", "--out-dir", required=True)
    fix.add_argument("--margin", type=int, default=16)
    fix.add_argument("--alpha-threshold", type=int, default=128)
    fix.set_defaults(func=_cmd_normalize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
