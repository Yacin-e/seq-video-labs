#!/usr/bin/env python3
"""Pose estimation with solvePnP (needs results/ from calibrate.py).

1. Draws 3D axes + a cube on every image (results/pose/).
2. Compares solvePnP poses with the (R, t) returned by calibrateCamera.
3. Subset experiment inspired by Quan & Lan (1999): pose from only n = 4, 5, 6... points vs. all points,
   with ITERATIVE / EPNP / IPPE (IPPE = planar-specific method), optional Gaussian pixel noise.

    python pose.py --results results
"""
import argparse
import csv
import os

import cv2
import numpy as np

from common import (get_plt, load_detections, load_image, pnp_pose, reproj_rms, rot_angle_deg)

METHODS = {"ITERATIVE": cv2.SOLVEPNP_ITERATIVE, "EPNP": cv2.SOLVEPNP_EPNP, "IPPE": cv2.SOLVEPNP_IPPE}


def draw_pose(img, K, dist, rvec, tvec, sq):
    th = max(2, img.shape[1] // 400)
    L = 3 * sq
    axes = np.float64([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, -L]])  # -Z points out of the board
    p = cv2.projectPoints(axes, rvec, tvec, K, dist)[0].reshape(-1, 2).astype(int)
    for i, col in zip((1, 2, 3), [(0, 0, 255), (0, 200, 0), (255, 0, 0)]):  # X red, Y green, Z blue
        cv2.line(img, tuple(p[0]), tuple(p[i]), col, th * 2, cv2.LINE_AA)
    s = 2 * sq
    base = np.float64([[0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0]])
    cube = np.vstack([base, base + [0, 0, -s]])
    q = cv2.projectPoints(cube, rvec, tvec, K, dist)[0].reshape(-1, 2).astype(int)
    cv2.polylines(img, [q[:4].reshape(-1, 1, 2)], True, (0, 255, 255), th, cv2.LINE_AA)
    cv2.polylines(img, [q[4:].reshape(-1, 1, 2)], True, (0, 255, 255), th, cv2.LINE_AA)
    for i in range(4):
        cv2.line(img, tuple(q[i]), tuple(q[i + 4]), (0, 255, 255), th, cv2.LINE_AA)


def subset_experiment(objp, corners, K, dist, refs, ns, noises, draws, rng):
    P = len(objp)
    rows = []
    for noise in noises:
        for n in ns:
            for mname, mflag in METHODS.items():
                rot_err, tr_err, fails, total = [], [], 0, 0
                for c, (Rref, tref) in zip(corners, refs):
                    c2 = c.reshape(-1, 2).astype(np.float64)
                    for _ in range(draws):
                        total += 1
                        idx = rng.choice(P, n, replace=False)
                        pts = c2[idx] + (rng.normal(0, noise, (n, 2)) if noise > 0 else 0)
                        try:
                            ok, rv, tv = cv2.solvePnP(objp[idx], pts.reshape(-1, 1, 2), K, dist, flags=mflag)
                        except cv2.error:
                            ok = False
                        # IPPE can return ok=True with a NaN pose when the points are (near-)collinear
                        if not ok or not (np.isfinite(rv).all() and np.isfinite(tv).all()):
                            fails += 1
                            continue
                        R = cv2.Rodrigues(rv)[0]
                        rot_err.append(rot_angle_deg(R, Rref))
                        tr_err.append(2 * np.linalg.norm(tv - tref) / (np.linalg.norm(tv) + np.linalg.norm(tref)))
                rows.append(dict(noise_px=noise, n_points=n, method=mname,
                                 rot_err_median_deg=float(np.median(rot_err)) if rot_err else np.nan,
                                 rot_err_mean_deg=float(np.mean(rot_err)) if rot_err else np.nan,
                                 trans_err_median=float(np.median(tr_err)) if tr_err else np.nan,
                                 trans_err_mean=float(np.mean(tr_err)) if tr_err else np.nan,
                                 failures=fails, trials=total))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--draws", type=int, default=100, help="random subsets per image / n / noise / method")
    ap.add_argument("--n-points", default="4,5,6,8,12,20")
    ap.add_argument("--noise", default="0,1,2,4", help="Gaussian pixel noise levels for the subset experiment")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-subsets", action="store_true")
    args = ap.parse_args()

    det = load_detections(args.results)
    cal = np.load(os.path.join(args.results, "calibration.npz"))
    K, dist = cal["K"], cal["dist"]
    objp, corners, files, sq = det["objp"], det["corners"], det["files"], det["square_mm"]
    cal_pose = {str(f): (cv2.Rodrigues(r)[0], t.reshape(3, 1))
                for f, r, t in zip(cal["files"], cal["rvecs"], cal["tvecs"])}
    os.makedirs(os.path.join(args.results, "pose"), exist_ok=True)

    rows, refs = [], []
    for f, c in zip(files, corners):
        name = os.path.basename(f)
        ok, rvec, tvec = pnp_pose(objp, c, K, dist)
        if not ok:
            print(f"  [skip] solvePnP failed on {name}")
            refs.append((np.eye(3), np.ones((3, 1))))
            continue
        R = cv2.Rodrigues(rvec)[0]
        refs.append((R, tvec))
        cam_pos = (-R.T @ tvec).ravel()
        tilt = float(np.degrees(np.arccos(abs(R[2, 2]))))
        row = dict(image=name, distance_mm=float(np.linalg.norm(tvec)), tilt_deg=tilt,
                   tx=float(tvec[0, 0]), ty=float(tvec[1, 0]), tz=float(tvec[2, 0]),
                   cam_x=float(cam_pos[0]), cam_y=float(cam_pos[1]), cam_z=float(cam_pos[2]),
                   pnp_reproj_rms_px=reproj_rms(objp, c, rvec, tvec, K, dist))
        if f in cal_pose:
            Rc, tc = cal_pose[f]
            row["rot_diff_vs_calib_deg"] = rot_angle_deg(R, Rc)
            row["trans_diff_vs_calib_mm"] = float(np.linalg.norm(tvec - tc))
        rows.append(row)
        img = load_image(f, det["max_width"])
        draw_pose(img, K, dist, rvec, tvec, sq)
        cv2.imwrite(os.path.join(args.results, "pose", os.path.splitext(name)[0] + "_pose.jpg"), img)

    keys = list(rows[0].keys())
    with open(os.path.join(args.results, "pose_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{'image':<26s}{'dist(mm)':>9s}{'tilt(deg)':>10s}{'PnP rms':>9s}{'drot(deg)':>10s}{'dt(mm)':>9s}")
    for r in rows:
        print(f"{r['image'][:25]:<26s}{r['distance_mm']:9.1f}{r['tilt_deg']:10.1f}{r['pnp_reproj_rms_px']:9.3f}"
              f"{r.get('rot_diff_vs_calib_deg', float('nan')):10.3f}{r.get('trans_diff_vs_calib_mm', float('nan')):9.2f}")
    print("\n(drot/dt ~ 0 is expected: solvePnP and calibrateCamera minimise the same reprojection error "
          "on the same points with the same K - it is a consistency check, not an independent validation.)")
    print(f"median |R_pnp - R_calib| ={np.median([r['rot_diff_vs_calib_deg'] for r in rows if 'rot_diff_vs_calib_deg' in r]):.3f} deg, "
          f"median |t_pnp - t_calib| = {np.median([r['trans_diff_vs_calib_mm'] for r in rows if 'trans_diff_vs_calib_mm' in r]):.2f} mm")

    if args.no_subsets:
        return
    ns = [int(v) for v in args.n_points.split(",")]
    noises = [float(v) for v in args.noise.split(",")]
    print("\nSubset experiment (reference = pose from all corners)...")
    rng = np.random.default_rng(args.seed)
    sub = subset_experiment(objp, corners, K, dist, refs, ns, noises, args.draws, rng)
    with open(os.path.join(args.results, "pose_subsets.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sub[0].keys()))
        w.writeheader()
        w.writerows(sub)
    print(f"{'noise':>6s}{'n':>4s}  {'method':<10s}{'rot med(deg)':>13s}{'trans med':>11s}{'fails':>7s}")
    for r in sub:
        print(f"{r['noise_px']:6.1f}{r['n_points']:4d}  {r['method']:<10s}{r['rot_err_median_deg']:13.4f}"
              f"{r['trans_err_median']:11.5f}{r['failures']:7d}")

    plt = get_plt()
    if plt:
        n0 = noises[0]
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
        for m in METHODS:
            sel = [r for r in sub if r["method"] == m and r["noise_px"] == n0]
            axs[0].plot([r["n_points"] for r in sel], [r["rot_err_median_deg"] for r in sel], "o-", label=m)
            axs[1].plot([r["n_points"] for r in sel], [r["trans_err_median"] for r in sel], "o-", label=m)
        for a, t in zip(axs, ["median rotation error (deg)", "median relative translation error"]):
            a.set_xlabel("number of points n")
            a.set_ylabel(t)
            a.set_yscale("log")
            a.grid(alpha=0.3)
        axs[0].legend()
        axs[0].set_title(f"vs n (noise = {n0:g} px)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.results, "fig_pose_vs_n.png"), dpi=200)
        plt.close(fig)

        if len(noises) > 1:
            fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
            for n, ls in ((4, "--"), (5, "-")):
                if n not in ns:
                    continue
                for m in METHODS:
                    sel = [r for r in sub if r["method"] == m and r["n_points"] == n]
                    axs[0].plot([r["noise_px"] for r in sel], [r["rot_err_median_deg"] for r in sel], "o" + ls, label=f"{m} n={n}")
                    axs[1].plot([r["noise_px"] for r in sel], [r["trans_err_median"] for r in sel], "o" + ls, label=f"{m} n={n}")
            for a, t in zip(axs, ["median rotation error (deg)", "median relative translation error"]):
                a.set_xlabel("added Gaussian noise (px)")
                a.set_ylabel(t)
                a.set_yscale("log")
                a.grid(alpha=0.3)
            axs[0].legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(os.path.join(args.results, "fig_pose_vs_noise.png"), dpi=200)
            plt.close(fig)
    print(f"\nSaved to '{args.results}/': pose/*.jpg, pose_table.csv, pose_subsets.csv, fig_pose_*.png")


if __name__ == "__main__":
    main()
