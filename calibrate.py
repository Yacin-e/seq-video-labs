#!/usr/bin/env python3
"""Camera calibration (chessboard) + focal length in meters via calibrationMatrixValues.

Example (iPhone XR photos in ./photos, 8x6 inner corners, 25 mm squares on your screen):
    python calibrate.py photos --pattern 8x6 --square-mm 25
"""
import argparse
import json
import os

import cv2
import numpy as np

from common import (IPHONE_XR_EXIF_F_MM, IPHONE_XR_SENSOR_MM, detect_all, get_plt, parse_pair,
                    run_calibration)

DIST_NAMES = ["k1", "k2", "p1", "p2", "k3"]


def distortion_map(K, dist, size, bounds=None):
    """Displacement (px) between distorted and undistorted positions on a grid.
    bounds = (x0, y0, x1, y1) restricts the grid, e.g. to the region where corners were observed."""
    w, h = size
    x0, y0, x1, y1 = bounds if bounds else (0, 0, w, h)
    xs, ys = np.meshgrid(np.linspace(x0, x1, 41), np.linspace(y0, y1, 31))
    pts = np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)
    mag = np.linalg.norm(und - pts.reshape(-1, 2), axis=1).reshape(xs.shape)
    return xs, ys, mag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", help="folder with the calibration photos")
    ap.add_argument("--pattern", default="8x6", help="INNER corners, cols x rows (default 8x6)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="side of one square in mm (measure it on the screen)")
    ap.add_argument("--sensor-mm", default=IPHONE_XR_SENSOR_MM, help="sensor width x height in mm (iPhone XR: 5.65x4.24)")
    ap.add_argument("--max-width", type=int, default=2016, help="downscale images to this width (0 = full resolution)")
    ap.add_argument("--max-err", type=float, default=None,
                    help="drop images whose reprojection RMS exceeds this (px) and recalibrate")
    ap.add_argument("--model", choices=["k1k2", "k1", "full"], default="k1k2",
                    help="distortion model: k1k2 = radial k1,k2 as in the course (default, stable); "
                         "k1 = single radial term; full = k1,k2,p1,p2,k3 (can overfit when the board "
                         "never reaches the image corners)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    flags = {"k1k2": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
             "k1": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
             "full": 0}[args.model]

    pattern = parse_pair(args.pattern, int)
    sensor_w, sensor_h = parse_pair(args.sensor_mm)
    os.makedirs(args.out, exist_ok=True)

    print(f"Detecting {pattern[0]}x{pattern[1]} inner corners...")
    det = detect_all(args.images, pattern, args.square_mm, args.max_width, args.out)
    files, corners, size, objp = det["files"], det["corners"], det["size"], det["objp"]
    print(f"\n{len(files)} usable images, processed size {size[0]}x{size[1]}")
    if len(files) < 8:
        print("  [warn] fewer than 8 images - take more (15-20 recommended)")

    # --- calibration (optionally prune bad views) ---
    dropped = []
    for _ in range(3):
        res = run_calibration(objp, corners, size, flags=flags)
        if args.max_err is None:
            break
        bad = [i for i, e in enumerate(res["per_view"]) if e > args.max_err]
        if not bad or len(files) - len(bad) < 6:
            break
        for i in sorted(bad, reverse=True):
            dropped.append((os.path.basename(files[i]), float(res["per_view"][i])))
            files.pop(i)
            corners.pop(i)
    K, dist, std = res["K"], res["dist"], res["std"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # --- focal length in mm / meters ---
    fovx, fovy, f_mm, pp_mm, aspect = cv2.calibrationMatrixValues(K, size, sensor_w, sensor_h)
    f_mm_y = fy * sensor_h / size[1]
    equiv35 = f_mm * 43.267 / np.hypot(sensor_w, sensor_h)
    pix_um = sensor_w / size[0] * 1000
    _, _, mag = distortion_map(K, dist, size)
    allc = np.concatenate([c.reshape(-1, 2) for c in corners])
    obs_bounds = (allc[:, 0].min(), allc[:, 1].min(), allc[:, 0].max(), allc[:, 1].max())
    _, _, mag_obs = distortion_map(K, dist, size, obs_bounds)

    lines = []
    P = lines.append
    P("=== CALIBRATION SUMMARY ===")
    P(f"images used            : {len(files)}   (dropped: {len(dropped)})")
    P(f"processed image size   : {size[0]} x {size[1]} px")
    P(f"pattern / square       : {pattern[0]}x{pattern[1]} inner corners, {args.square_mm} mm")
    P(f"distortion model       : {args.model}")
    P(f"RMS reprojection error : {res['rms']:.4f} px")
    P("")
    P("Intrinsic matrix K (pixels):")
    for r in K:
        P("   [" + "  ".join(f"{v:10.3f}" for v in r) + " ]")
    P(f"fx = {fx:.2f} +/- {std[0]:.2f}   fy = {fy:.2f} +/- {std[1]:.2f}   (px, +/- = std dev)")
    P(f"cx = {cx:.2f} +/- {std[2]:.2f}   cy = {cy:.2f} +/- {std[3]:.2f}   "
      f"(image centre: {(size[0] - 1) / 2:.1f}, {(size[1] - 1) / 2:.1f})")
    P(f"aspect fx/fy           : {fx / fy:.5f}")
    P("Distortion coefficients: " + "  ".join(f"{n}={v:+.5f}" for n, v in zip(DIST_NAMES, dist)))
    P(f"max distortion displacement, observed region : {mag_obs.max():.2f} px  "
      f"(bbox x {obs_bounds[0]:.0f}-{obs_bounds[2]:.0f}, y {obs_bounds[1]:.0f}-{obs_bounds[3]:.0f})")
    P(f"max distortion displacement, image corners   : {mag.max():.2f} px  "
      f"(extrapolated if the board never reached the corners)")
    P("")
    P(f"--- calibrationMatrixValues (sensor {sensor_w} x {sensor_h} mm) ---")
    P(f"FOV x / y              : {fovx:.2f} deg / {fovy:.2f} deg")
    P(f"focal length (from fx) : {f_mm:.4f} mm   = {f_mm / 1000:.6f} m")
    P(f"focal length (from fy) : {f_mm_y:.4f} mm   = {f_mm_y / 1000:.6f} m")
    P(f"principal point        : ({pp_mm[0]:.3f}, {pp_mm[1]:.3f}) mm from the sensor corner")
    P(f"aspect ratio           : {aspect:.5f}")
    P(f"35 mm equivalent       : {equiv35:.1f} mm")
    P(f"effective pixel size   : {pix_um:.3f} um at the processed resolution")
    P(f"reference (iPhone XR EXIF nominal): {IPHONE_XR_EXIF_F_MM} mm  -> difference {100 * (f_mm / IPHONE_XR_EXIF_F_MM - 1):+.1f} %")
    P("")
    P("Per-image reprojection RMS (px):")
    for f, e in zip(files, res["per_view"]):
        P(f"   {os.path.basename(f):<30s} {e:.3f}")
    if dropped:
        P("Dropped images (RMS > --max-err): " + ", ".join(f"{n} ({e:.2f})" for n, e in dropped))
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")

    summary = dict(
        n_images=len(files), image_size=list(size), pattern=list(pattern), square_mm=args.square_mm,
        distortion_model=args.model,
        sensor_mm=[sensor_w, sensor_h], rms_px=res["rms"], K=K.tolist(), fx=fx, fy=fy, cx=cx, cy=cy,
        std_fx_fy_cx_cy=[float(v) for v in std[:4]], dist=dict(zip(DIST_NAMES, map(float, dist))),
        fov_x_deg=fovx, fov_y_deg=fovy, focal_mm_from_fx=f_mm, focal_m_from_fx=f_mm / 1000,
        focal_mm_from_fy=f_mm_y, focal_m_from_fy=f_mm_y / 1000, aspect_ratio=aspect,
        equiv_35mm=equiv35, max_distortion_px_image_corners=float(mag.max()),
        max_distortion_px_observed_region=float(mag_obs.max()),
        per_image_rms={os.path.basename(f): float(e) for f, e in zip(files, res["per_view"])},
        dropped=dropped)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez(os.path.join(args.out, "calibration.npz"), K=K, dist=dist, image_size=np.array(size),
             files=np.array(files), rvecs=np.array(res["rvecs"]), tvecs=np.array(res["tvecs"]),
             square_mm=args.square_mm, pattern=np.array(pattern), sensor_mm=np.array([sensor_w, sensor_h]))
    # keep the detections cache consistent with the images actually used
    np.savez(os.path.join(args.out, "detections.npz"), files=np.array(files), corners=np.array(corners),
             image_size=np.array(size), pattern=np.array(pattern), square_mm=args.square_mm,
             max_width=args.max_width or 0)

    # --- undistortion before / after ---
    from common import load_image
    img = load_image(files[0], args.max_width)
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, 1)
    und = cv2.undistort(img, K, dist, None, newK)
    sbs = np.hstack([img, und])
    s = 1800 / sbs.shape[1]
    sbs = cv2.resize(sbs, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.putText(sbs, "original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.putText(sbs, "undistorted", (sbs.shape[1] // 2 + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite(os.path.join(args.out, "undistort_before_after.jpg"), sbs)

    # --- figures ---
    plt = get_plt()
    if plt:
        names = [os.path.basename(f) for f in files]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(range(len(names)), res["per_view"], color="#4C72B0")
        ax.axhline(res["rms"], color="#C44E52", ls="--", label=f"overall RMS = {res['rms']:.3f} px")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n[:10] for n in names], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("reprojection RMS (px)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fig_per_image_error.png"), dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        allc = np.concatenate([c.reshape(-1, 2) for c in corners])
        ax.scatter(allc[:, 0], allc[:, 1], s=3, alpha=0.5)
        ax.set_xlim(0, size[0])
        ax.set_ylim(size[1], 0)
        ax.set_aspect("equal")
        ax.set_title("Corner coverage over all images")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fig_coverage.png"), dpi=200)
        plt.close(fig)

        xs, ys, mag = distortion_map(K, dist, size)
        fig, ax = plt.subplots(figsize=(5.8, 4.3))
        cs = ax.contourf(xs, ys, mag, 14, cmap="viridis")
        fig.colorbar(cs, label="displacement (px)")
        ax.plot(cx, cy, "r+", ms=12)
        ax.set_ylim(size[1], 0)
        ax.set_aspect("equal")
        ax.set_title("Radial/tangential distortion magnitude")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "fig_distortion_map.png"), dpi=200)
        plt.close(fig)

    print(f"\nSaved to '{args.out}/': summary.txt, summary.json, calibration.npz, undistort_before_after.jpg, fig_*.png, corners/")


if __name__ == "__main__":
    main()
