"""Сводка по всем комплектам: SSIM и Pearson по плоскостям и средние.

Читает ``metrics.json`` каждого комплекта в ``data/sets/`` (и сдаваемый набор
``outputs/metrics.json``), сводит в одну таблицу, пишет ``outputs/sets_summary.json``
и печатает markdown-таблицу — её же можно вставить в README.

    python scripts/report_sets.py
    python scripts/report_sets.py --md outputs/sets_summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hdoe import sets as SS     # noqa: E402


def rows() -> list[dict]:
    out = []
    # основной результат: сдаваемый набор
    main = os.path.join(ROOT, "outputs", "metrics.json")
    if os.path.exists(main):
        m = json.load(open(main))
        planes = m.get("planes") or []
        out.append(dict(id="data/targets", kind="сдаваемый набор", source="outputs/metrics.json",
                        ssim=float(np.mean([p["ssim"] for p in planes])) if planes else None,
                        pcc=float(np.mean([p["pcc"] for p in planes])) if planes else None,
                        ssim_planes=[float(p["ssim"]) for p in planes],
                        pcc_planes=[float(p["pcc"]) for p in planes],
                        ssim_raw=float(np.mean([p.get("ssim_raw", np.nan) for p in planes])),
                        steps=(m.get("phys") or {}).get("steps")))
    for info in SS.all_sets():
        sid = info["id"]
        if sid == "default":            # копия сдаваемого набора, в таблице лишняя
            continue
        path = os.path.join(SS.set_path(sid), "metrics.json")
        if not os.path.exists(path):
            continue
        m = json.load(open(path))
        planes = m.get("planes") or []
        out.append(dict(id=sid, kind=(info.get("kind") or ("загружен через дашборд"
                                                           if not info.get("synthetic")
                                                           else "комплект")),
                        source=os.path.relpath(path, ROOT),
                        ssim=float(np.mean([p["ssim"] for p in planes])),
                        pcc=float(np.mean([p["pearson"] for p in planes])),
                        ssim_planes=[float(p["ssim"]) for p in planes],
                        pcc_planes=[float(p["pearson"]) for p in planes],
                        ssim_raw=float(np.mean([p.get("ssim_raw", np.nan) for p in planes])),
                        steps=m.get("steps"),
                        seconds=m.get("seconds")))
    return out


def markdown(data: list[dict]) -> str:
    head = ("| комплект | тип | SSIM | Pearson | SSIM z₁…z₄ |\n"
            "|---|---|---:|---:|---|\n")
    body = ""
    for r in data:
        planes = " / ".join(f"{v:.3f}" for v in r["ssim_planes"])
        body += (f"| `{r['id']}` | {r['kind']} | **{r['ssim']:.3f}** | "
                 f"**{r['pcc']:.3f}** | {planes} |\n")
    ssim = [r["ssim"] for r in data if r["ssim"] is not None]
    pcc = [r["pcc"] for r in data if r["pcc"] is not None]
    tail = (f"\nСреднее по всем комплектам: SSIM {np.mean(ssim):.3f}, "
            f"Pearson {np.mean(pcc):.3f}.\n")
    return head + body + tail


def main() -> int:
    ap = argparse.ArgumentParser(description="Сводка метрик по комплектам")
    ap.add_argument("--md", default=os.path.join(ROOT, "outputs", "sets_summary.md"))
    args = ap.parse_args()
    data = rows()
    if not data:
        print("нет ни одного metrics.json — сначала прогоните scripts/evaluate.py "
              "или scripts/make_sample_sets.py --infer")
        return 1
    os.makedirs(os.path.dirname(args.md), exist_ok=True)
    with open(args.md, "w") as fh:
        fh.write(markdown(data))
    with open(os.path.join(ROOT, "outputs", "sets_summary.json"), "w") as fh:
        json.dump(dict(sets=data, created=__import__("time").strftime("%Y-%m-%dT%H:%M:%S")),
                  fh, indent=1, ensure_ascii=False)
    print(markdown(data))
    print(f"записано: outputs/sets_summary.json, {os.path.relpath(args.md, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
