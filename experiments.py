#!/usr/bin/env python3
"""Calibration experiments for the analysis section (needs results/ from calibrate.py).

A. Effect of the number of images (random subsets; remaining images = validation set).
B. Effect of the camera model: full distortion / k1,k2 only (course model) / none /
   fixed aspect ratio / fixed principal point. Held-out reprojection error from random 70/30 splits.

    python experiments.py --results results
"""
import argparse
import csv
import os
import warnings

import cv2
import numpy as np

from common import (IPHONE_XR_SENSOR_MM, get_plt, load_detections, parse_pair, pnp_pose, reproj_rms,
                    run_calibration)

CONFIGS = {
    "full (k1,k2,p1,p2,k3)": dict(flags=0),
    "k1,k2 only (course model)": dict(flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3),
    "no distortion": dict(flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3),
    "full + fixed aspect ratio": dict(flags=cv2.CALIB_FIX_ASPECT_RATIO, K0=np.eye(3)),
    "full + fixed principal point": dict(flags=cv2.CALIB_FIX_PRINCIPAL_POINT),
}


def val_rms(objp, corners, idx, K, dist):
    errs = []
    for i in idx:
        ok, rv, tv = pnp_pose(objp, corners[i], K, dist)
        if ok:
            errs.append(reproj_rms(objp, corners[i], rv, tv, K, dist))
    return float(np.mean(errs)) if errs else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--sensor-mm", default=IPHONE_XR_SENSOR_MM)
    ap.add_argument("--repeats", type=int, default=30, help="random subsets per n (experiment A)")
    ap.add_argument("--splits", type=int, default=20, help="random 70/30 splits per model (experiment B)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN held-out slice when n == N

    det = load_detections(args.results)
    objp, corners, size = det["objp"], det["corners"], det["size"]
    N = len(corners)
    sw, sh = parse_pair(args.sensor_mm)
    rng = np.random.default_rng(args.seed)
    plt = get_plt()
    print(f"{N} images, size {size[0]}x{size[1]}")

    # ---------------- A: number of images ----------------
    ns = [n for n in (3, 4, 5, 6, 8, 10, 12, 15, 20, 25) if n < N] + [N]
    recA, aggA = [], []
    for n in ns:
        reps = 1 if n == N else args.repeats
        cur = []
        for _ in range(reps):
            idx = rng.choice(N, n, replace=False)
            try:
                res = run_calibration(objp, [corners[i] for i in idx], size)
            except cv2.error:
                continue
            held = [i for i in range(N) if i not in set(idx)]
            K = res["K"]
            row = dict(n_images=n, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                       f_mm=K[0, 0] * sw / size[0], k1=res["dist"][0], rms_train=res["rms"],
                       rms_heldout=val_rms(objp, corners, held, K, res["dist"]) if held else np.nan)
            cur.append(row)
            recA.append(row)
        agg = {k: (float(np.nanmean([r[k] for r in cur])), float(np.nanstd([r[k] for r in cur])))
               for k in ("fx", "fy", "cx", "cy", "f_mm", "k1", "rms_train", "rms_heldout")}
        agg["n_images"] = n
        aggA.append(agg)
    with open(os.path.join(args.results, "exp_num_images.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recA[0].keys()))
        w.writeheader()
        w.writerows(recA)
    print("\nA. Number of images (mean +/- std over random subsets)")
    print(f"{'n':>3s} {'fx':>16s} {'cx':>14s} {'f (mm)':>14s} {'rms train':>13s} {'rms held-out':>14s}")
    for a in aggA:
        print(f"{a['n_images']:3d} {a['fx'][0]:9.1f}+/-{a['fx'][1]:5.1f} {a['cx'][0]:8.1f}+/-{a['cx'][1]:4.1f} "
              f"{a['f_mm'][0]:8.3f}+/-{a['f_mm'][1]:5.3f} {a['rms_train'][0]:7.3f}+/-{a['rms_train'][1]:4.2f} "
              f"{a['rms_heldout'][0]:8.3f}+/-{a['rms_heldout'][1]:4.2f}")
    if plt:
        x = [a["n_images"] for a in aggA]
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.3))
        for ax, key, lab in zip(axs, ("fx", "cx", "rms_heldout"), ("fx (px)", "cx (px)", "held-out reprojection RMS (px)")):
            ax.errorbar(x, [a[key][0] for a in aggA], yerr=[a[key][1] for a in aggA], fmt="o-", capsize=3)
            ax.set_xlabel("number of calibration images")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.results, "fig_exp_num_images.png"), dpi=200)
        plt.close(fig)

    # ---------------- B: camera model ----------------
    print("\nB. Camera model (fit on all images; held-out RMS from random 70/30 splits)")
    ntr = max(4, int(round(0.7 * N)))
    recB = []
    print(f"{'model':<30s}{'RMS':>7s}{'fx':>9s}{'fy':>9s}{'cx':>8s}{'cy':>8s}{'f(mm)':>8s}{'k1':>9s}{'k2':>9s}{'held-out':>10s}")
    for name, cfg in CONFIGS.items():
        K0 = cfg.get("K0")
        res = run_calibration(objp, corners, size, flags=cfg["flags"], K0=None if K0 is None else K0.copy())
        K, d = res["K"], res["dist"]
        hv = []
        for _ in range(args.splits):
            perm = rng.permutation(N)
            tr, te = perm[:ntr], perm[ntr:]
            try:
                r2 = run_calibration(objp, [corners[i] for i in tr], size, flags=cfg["flags"],
                                     K0=None if K0 is None else K0.copy())
            except cv2.error:
                continue
            hv.append(val_rms(objp, corners, te, r2["K"], r2["dist"]))
        row = dict(model=name, rms=res["rms"], fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                   f_mm=K[0, 0] * sw / size[0], k1=d[0], k2=d[1], p1=d[2], p2=d[3], k3=d[4],
                   heldout_mean=float(np.nanmean(hv)), heldout_std=float(np.nanstd(hv)))
        recB.append(row)
        print(f"{name:<30s}{row['rms']:7.3f}{row['fx']:9.1f}{row['fy']:9.1f}{row['cx']:8.1f}{row['cy']:8.1f}"
              f"{row['f_mm']:8.3f}{row['k1']:9.4f}{row['k2']:9.4f}{row['heldout_mean']:8.3f}+/-{row['heldout_std']:.2f}")
    with open(os.path.join(args.results, "exp_models.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recB[0].keys()))
        w.writeheader()
        w.writerows(recB)
    if plt:
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
        xs = np.arange(len(recB))
        ax.bar(xs - 0.2, [r["rms"] for r in recB], 0.4, label="train RMS (all images)")
        ax.bar(xs + 0.2, [r["heldout_mean"] for r in recB], 0.4, yerr=[r["heldout_std"] for r in recB],
               capsize=3, label="held-out RMS (70/30 splits)")
        ax.set_xticks(xs)
        ax.set_xticklabels([r["model"].replace(" (", "\n(").replace(" + ", "\n+ ") for r in recB], fontsize=7)
        ax.set_ylabel("reprojection RMS (px)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(args.results, "fig_exp_models.png"), dpi=200)
        plt.close(fig)
    print(f"\nSaved to '{args.results}/': exp_num_images.csv, exp_models.csv, fig_exp_*.png")


if __name__ == "__main__":
    main()
