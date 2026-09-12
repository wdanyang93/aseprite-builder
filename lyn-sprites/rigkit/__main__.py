"""CLI — 하드코딩 경로 없음. 모든 입출력은 인자로 받는다."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m rigkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cut", help="파츠 시트를 조각으로 자른다")
    c.add_argument("sheet"); c.add_argument("-o", "--out", required=True)
    c.add_argument("--min-area", type=int, default=400)
    c.add_argument("--tol", type=int, default=18)
    c.add_argument("--gap", type=int, default=5)

    a = sub.add_parser("assign", help="조각에 슬롯 이름과 앵커를 붙인다")
    a.add_argument("pieces"); a.add_argument("names"); a.add_argument("-o", "--out", required=True)

    p = sub.add_parser("preview", help="걷기 스트립/GIF 를 굽는다")
    p.add_argument("parts"); p.add_argument("-o", "--out", required=True)
    p.add_argument("--side", default="east"); p.add_argument("--frames", type=int, default=8)
    p.add_argument("--fps", type=int, default=12)

    g = sub.add_parser("godot", help="Godot 4 씬(.tscn) 을 내보낸다")
    g.add_argument("parts"); g.add_argument("-o", "--out", required=True)
    g.add_argument("--side", default="east"); g.add_argument("--frames", type=int, default=8)
    g.add_argument("--fps", type=int, default=12)
    g.add_argument("--tex-root", default="res://assets/parts")

    f = sub.add_parser("fit", help="파츠 크기를 리그 뼈 길이에 맞춘다 (parts.json 의 scale 갱신)")
    f.add_argument("parts"); f.add_argument("--dry-run", action="store_true")

    ck = sub.add_parser("check", help="시트 안에서 파츠끼리 비율이 맞는지 본다")
    ck.add_argument("parts")

    sp = sub.add_parser("split", help="통짜 팔다리를 관절에서 두 조각으로 나눈다")
    sp.add_argument("image"); sp.add_argument("upper"); sp.add_argument("lower")
    sp.add_argument("-o", "--out", required=True)
    sp.add_argument("--at", type=float, default=0.5)
    sp.add_argument("--overlap", type=int, default=12)

    ph = sub.add_parser("placeholder", help="검증용 임시 파츠를 만든다")
    ph.add_argument("-o", "--out", required=True); ph.add_argument("--side", default="east")

    ns = ap.parse_args(argv)

    if ns.cmd == "cut":
        from .cut import cut
        m = cut(ns.sheet, ns.out, min_area=ns.min_area, tol=ns.tol, gap=ns.gap)
        print(f"조각 {len(m['pieces'])}개 → {ns.out}  (CONTACT.png 를 보고 names.json 을 적는다)")
    elif ns.cmd == "assign":
        from .assign import assign
        parts = assign(ns.pieces, ns.names, ns.out)
        print(f"슬롯 {len(parts)}개 → {Path(ns.out) / 'parts.json'}")
    elif ns.cmd == "preview":
        from .render import load_parts, render_cycle, save_gif, strip
        from .rig import build_skeleton
        from .walk import walk_cycle
        out = Path(ns.out); out.mkdir(parents=True, exist_ok=True)
        fr = render_cycle(load_parts(ns.parts), build_skeleton(ns.side),
                          walk_cycle(ns.frames, side=ns.side))
        for i, f in enumerate(fr):
            f.save(out / f"{i}.png")
        strip(fr, cols=len(fr)).save(out / "STRIP.png")
        save_gif(fr, out / "WALK.gif", fps=ns.fps)
        print(f"{len(fr)}프레임 → {out}/STRIP.png, WALK.gif")
    elif ns.cmd == "godot":
        from .godot import export_scene
        p = export_scene(ns.parts, ns.out, side=ns.side, fps=ns.fps,
                         frames=ns.frames, tex_root=ns.tex_root)
        print(f"→ {p}")
    elif ns.cmd == "fit":
        from .fit import fit_parts
        r = fit_parts(ns.parts, write=not ns.dry_run)
        for slot, v in sorted(r.items(), key=lambda kv: -abs(kv[1]["scale"] - 1)):
            print(f'{slot:12s} 잰길이 {v["measured"]:7.1f}  기준 {v["target"]:7.1f}  배율 {v["scale"]:.3f}')
        print("(dry-run: 파일을 쓰지 않았다)" if ns.dry_run else f'→ {Path(ns.parts) / "parts.json"} 갱신')
    elif ns.cmd == "check":
        from .fit import consistency
        c = consistency(ns.parts)
        print(f'시트 전체 배율(중앙값) {c["median_scale"]:.3f}')
        bad = {k: v for k, v in c["deviation_pct"].items() if abs(v) > 8}
        if bad:
            print("비율이 어긋난 파츠(중앙값 대비 %):")
            for k, v in sorted(bad.items(), key=lambda kv: -abs(kv[1])):
                print(f"  {k:12s} {v:+.1f}%")
            print("→ 8% 넘게 어긋나면 그 파츠만 AI 에게 다시 요구하는 편이 낫다.")
        else:
            print("파츠끼리 비율이 맞는다 (모두 ±8% 이내).")
    elif ns.cmd == "split":
        from .split import split_to_files
        print(split_to_files(ns.image, ns.out, ns.upper, ns.lower,
                             at=ns.at, overlap=ns.overlap))
    elif ns.cmd == "placeholder":
        from .placeholder import build
        print(f"→ {build(ns.out, ns.side)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
