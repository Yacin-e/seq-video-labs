"""Shared helpers for the camera calibration scripts (TP Analyse de Sequences Video)."""
import glob
import os
import sys
from collections import Counter

import cv2
import numpy as np

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".heif")

# iPhone XR main camera: 1/2.55" sensor, 1.4 um pixels, 4032x3024 -> ~5.65 x 4.24 mm.
# Nominal focal length in the EXIF: 4.25 mm. Used as defaults / sanity check.
IPHONE_XR_SENSOR_MM = "5.65x4.24"
IPHONE_XR_EXIF_F_MM = 4.25


def parse_pair(text, cast=float):
    a, b = text.lower().replace("\u00d7", "x").split("x")
    return cast(a), cast(b)


def list_images(folder):
    files = sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(EXTS))
    if not files:
        sys.exit(f"No images found in '{folder}'")
    return files


def load_image(path, max_width=0):
    """Load an image as BGR. EXIF orientation is applied. Optionally downscale to max_width."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".heic", ".heif"):
        try:
            import pillow_heif
            from PIL import Image, ImageOps
            pillow_heif.register_heif_opener()
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        except ImportError:
            sys.exit("HEIC images need:  pip install pillow-heif   "
                     "(or set iPhone Settings > Camera > Formats > Most Compatible, or export as JPG)")
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # applies EXIF orientation
    if img is None:
        return None
    h, w = img.shape[:2]
    if max_width and w > max_width:
        s = max_width / w
        img = cv2.resize(img, (max_width, int(round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def object_points(pattern, square_mm, square_mm_y=None):
    """3D coordinates (mm) of the inner corners in the board frame (Z = 0).
    square_mm = square width (along the columns), square_mm_y = square height (default: same)."""
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * [square_mm, square_mm_y or square_mm]
    return objp


def detect_corners(gray, pattern):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if ok:
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    if hasattr(cv2, "findChessboardCornersSB"):  # more robust fallback (already sub-pixel)
        ok, corners = cv2.findChessboardCornersSB(
            gray, pattern, cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ok:
            return corners
    return None


def detect_all(folder, pattern, square_mm, max_width, out_dir, square_mm_y=None):
    """Detect the chessboard in every image, save debug images + a cache (detections.npz)."""
    files = list_images(folder)
    os.makedirs(os.path.join(out_dir, "corners"), exist_ok=True)
    found = []  # (path, size, corners)
    for f in files:
        name = os.path.basename(f)
        img = load_image(f, max_width)
        if img is None:
            print(f"  [skip] {name}: cannot read")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = detect_corners(gray, pattern)
        if c is None:
            print(f"  [skip] {name}: chessboard {pattern[0]}x{pattern[1]} (inner corners) not found")
            continue
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern, c, True)
        cv2.imwrite(os.path.join(out_dir, "corners", os.path.splitext(name)[0] + ".jpg"), vis)
        found.append((os.path.abspath(f), (img.shape[1], img.shape[0]), c.reshape(-1, 1, 2).astype(np.float32)))
        print(f"  [ ok ] {name}")
    if not found:
        sys.exit("No chessboard detected in any image. Check --pattern (INNER corners, cols x rows).")
    size = Counter(s for _, s, _ in found).most_common(1)[0][0]
    kept = [(p, c) for p, s, c in found if s == size]
    if len(kept) < len(found):
        print(f"  [warn] {len(found) - len(kept)} image(s) dropped: size differs from {size[0]}x{size[1]} "
              f"(mixed portrait/landscape or zoom?)")
    np.savez(os.path.join(out_dir, "detections.npz"),
             files=np.array([p for p, _ in kept]), corners=np.array([c for _, c in kept]),
             image_size=np.array(size), pattern=np.array(pattern), square_mm=square_mm,
             square_mm_y=square_mm_y or square_mm, max_width=max_width or 0)
    return load_detections(out_dir)


def load_detections(out_dir):
    path = os.path.join(out_dir, "detections.npz")
    if not os.path.exists(path):
        sys.exit(f"{path} not found - run calibrate.py first.")
    d = np.load(path, allow_pickle=False)
    pattern = tuple(int(v) for v in d["pattern"])
    square = float(d["square_mm"])
    square_y = float(d["square_mm_y"]) if "square_mm_y" in d.files else square
    return dict(files=[str(f) for f in d["files"]], corners=[c for c in d["corners"]],
                size=tuple(int(v) for v in d["image_size"]), pattern=pattern, square_mm=square,
                square_mm_y=square_y, max_width=int(d["max_width"]),
                objp=object_points(pattern, square, square_y))


def run_calibration(objp, corners, size, flags=0, K0=None):
    obj = [objp.astype(np.float32)] * len(corners)
    rms, K, dist, rvecs, tvecs, std_int, _, per_view = cv2.calibrateCameraExtended(
        obj, [c.astype(np.float32) for c in corners], size, K0, None, flags=flags)
    return dict(rms=float(rms), K=K, dist=dist.ravel(), rvecs=rvecs, tvecs=tvecs,
                std=std_int.ravel(), per_view=per_view.ravel())


def pnp_pose(objp, c, K, dist, flags=cv2.SOLVEPNP_ITERATIVE):
    return cv2.solvePnP(objp, c.reshape(-1, 1, 2).astype(np.float64), K, dist, flags=flags)


def reproj_rms(objp, c, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    d = proj.reshape(-1, 2) - c.reshape(-1, 2)
    return float(np.sqrt((d ** 2).sum(1).mean()))


def rot_angle_deg(R1, R2):
    R = R1 @ R2.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("  [info] matplotlib not installed -> figures skipped (pip install matplotlib)")
        return None
