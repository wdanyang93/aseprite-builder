"""python -m bench run <case|group|all> [--methods a,b] [--jobs N] [--fresh] [--refit] [--no-sheets]
   python -m bench list                 cases and methods
   python -m bench t0                   T0 reproduction table (<out>/T0.md)
   python -m bench azimuth              azimuth estimators on synthetic truth (<out>/AZIMUTH.md)
`run all` also runs `azimuth` and `t0`. Paths: bench/paths.local.json (copy paths.example.json)."""
import argparse
import os
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("spec", help="case name, group (real, synth, recolour, small, frames), prefix, comma list, or all")
    r.add_argument("--methods", default=None, help="comma list (default: all)")
    r.add_argument("--jobs", type=int, default=2)
    r.add_argument("--fresh", action="store_true", help="recompute outputs and metrics (keep fitted models)")
    r.add_argument("--refit", action="store_true", help="refit models too")
    r.add_argument("--no-sheets", action="store_true")
    r.add_argument("--no-extra", action="store_true", help="with 'all': skip azimuth + T0")
    sub.add_parser("list")
    t = sub.add_parser("t0")
    t.add_argument("--jobs", type=int, default=2)
    z = sub.add_parser("azimuth")
    z.add_argument("--jobs", type=int, default=2)
    a = p.parse_args(argv)
    jobs = getattr(a, "jobs", 1)
    # keep BLAS / OpenCV threads in proportion to the worker count (4 shared CPUs)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, str(max(1, 4 // max(jobs, 1))))

    from . import cases as CS, methods as MT, config
    if a.cmd == "list":
        for k, c in CS.get().items():
            print(f"{k:<26} {c.group:<9} fit={c.fit_id:<34} frames={len(c.frames):<3} {c.desc}")
        print("methods:", ", ".join(MT.DEFAULT))
        return 0
    if a.cmd == "t0":
        from . import t0
        t0.main(jobs=a.jobs)
        return 0
    if a.cmd == "azimuth":
        from . import estimators
        estimators.main(jobs=a.jobs)
        return 0
    from . import run as RUN, sheets
    names = CS.select(a.spec)
    ms = a.methods.split(",") if a.methods else list(MT.DEFAULT)
    rows = RUN.run(names, ms, jobs=a.jobs, fresh=a.fresh, refit=a.refit)
    agg = RUN.aggregate(rows, names, ms)
    table = RUN.fmt_table(agg)
    base = os.path.join(config.out_dir(), "results_" + ("all" if a.spec == "all" else MT._safe(a.spec)))
    RUN.write_results(rows, agg, base)
    with open(base + ".txt", "w") as f:
        f.write(table + "\n")
    if not a.no_sheets:
        sheets.write_all(names, ms)
    print(table)
    nan_rows = [x for x in agg if x["nan"]]
    errs = [r for r in rows if str(r.get("status", "")).startswith(("ERROR", "FIT ERROR"))]
    print(f"\nresults -> {base}.tsv / .json / .txt   sheets -> {config.out_dir('sheets')}")
    print(f"NaN guard: {len(nan_rows)} case x method rows with NaN metrics"
          + ("".join(f"\n  {x['case']} {x['method']}: {', '.join(x['nan'])}" for x in nan_rows[:40])))
    print(f"errors: {len(errs)}" + "".join(f"\n  {e['case']} {e['frame']} {e['method']}: {e.get('note', '')[:120]}" for e in errs[:40]))
    if a.spec == "all" and not a.no_extra:
        from . import estimators, t0
        estimators.main(jobs=a.jobs)
        t0.main(jobs=a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
