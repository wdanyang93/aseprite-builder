# bench - shared test bench for the general relight method (B0)

One place that runs every method on every case with one set of metrics.

```bash
cp bench/paths.example.json bench/paths.local.json   # fill in: private data root + output folder (gitignored)
python -m bench list                                  # cases and methods
python -m bench run all                               # every case x method, one table, sheets, AZIMUTH.md, T0.md
python -m bench run cross_RL,P01_night_rgba --methods tool_profile,M4v3
python -m bench run synth --jobs 3                    # a group: real, synth, recolour, small, frames
python -m bench t0                                    # reproduce the research numbers (T0.md)
python -m bench azimuth                               # azimuth estimators on synthetic truth (AZIMUTH.md)
```

Nothing is written into the repo: models, outputs, metrics, sheets and the generated held-out frames go to `out`
(from `paths.local.json`). Delete `out/cache/fits` to refit, `out/cache/out` to re-apply; metrics are recomputed
automatically when `metrics.py` changes. User art never enters the repo; prototypes are imported from the configured
research folders (`protos.py`), not copied. The renderers (`synth_*.py`) are copies of our own research renderers.

| file | what |
|---|---|
| `metrics.py` | the unified metrics (dEa, dEst, rL, sun/shd edge bands, con, det, lnC, hp, alpha, lightened lines, NaN guard) + compat replicas of the research metrics |
| `parts.py` | frozen body-part polygons of the sun-rim benchmark |
| `cases.py` | case registry (real pairs, synthetic scenes x 3 renderers x 13 drawings, recolours, small frames, pseudo frames) |
| `frames.py` | pseudo frames; the NEW held-out set (seeds 7101-7303 right, 8101-8303 left) |
| `methods.py` | adapters: tool_profile, tool_pack, M4v3, M2av3, proto_ratio, model2 - add a `Method` subclass for a new one |
| `run.py`, `sheets.py`, `__main__.py` | runner (cached, parallel), table, sheets, CLI |
| `estimators.py`, `t0.py` | azimuth estimator comparison, T0 reproduction |

Angle convention everywhere: degrees, 0 = screen right, 90 = screen up (image rows grow down: direction `deg` is the
image vector `(cos deg, -sin deg)`). Lengths are fractions of the figure height H; the sun/shadow edge bands are
0-2 / 2-4 px at H = 1392 px, scaled with the frame (never thinner than 1 px).
