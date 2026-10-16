#!/usr/bin/env python3
"""
vision_geolocation_demo.py
==========================
Standalone, offline demo:  aerial photo -> YOLOv8 person detection ->
pixel (u, v) -> local ground coordinates (gx, gy) via the pinhole model
from the GCS simulation (pranav2.py).

    pip install ultralytics opencv-python numpy

Usage
-----
    python vision_geolocation_demo.py                       # defaults
    python vision_geolocation_demo.py --image aerial_sample.jpg --alt 80 --yaw-deg 0
    python vision_geolocation_demo.py --tile 640            # sliced inference (SAHI-style)

Why CameraSpecs / GeolocationEngine are *copied* here
-----------------------------------------------------
pranav2.py calls matplotlib.use("TkAgg") and does `from stl import mesh` at
import time, so `import pranav2` fails on headless machines or without
numpy-stl. The two classes below are the same math (one dead line removed),
so this file runs anywhere with only numpy + opencv + ultralytics.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

PERSON_CLASS_ID = 0  # COCO class 0 = "person"


# ════════════════════════════════════════════════════════════════════════
#  1.  CAMERA + GEOLOCATION  (same model as pranav2.py, Modules 0-B & 4)
# ════════════════════════════════════════════════════════════════════════
@dataclass
class CameraSpecs:
    """Nadir-pointing pinhole camera (1/2.3" sensor, 4.5 mm lens)."""
    sensor_w_mm: float = 6.17
    sensor_h_mm: float = 4.55
    focal_mm: float = 4.5
    res_x: int = 4000
    res_y: int = 3000

    def __post_init__(self):
        self.fx = self.focal_mm * self.res_x / self.sensor_w_mm
        self.fy = self.focal_mm * self.res_y / self.sensor_h_mm
        self.cx = self.res_x / 2.0
        self.cy = self.res_y / 2.0
        self.K = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fy, self.cy],
                           [0.0, 0.0, 1.0]])
        self.K_inv = np.linalg.inv(self.K)
        self.hfov_rad = 2.0 * np.arctan(self.sensor_w_mm / (2.0 * self.focal_mm))
        self.vfov_rad = 2.0 * np.arctan(self.sensor_h_mm / (2.0 * self.focal_mm))

    def ground_footprint(self, altitude: float) -> Tuple[float, float]:
        """(width_m, height_m) of the image footprint on flat ground."""
        return (2.0 * altitude * np.tan(self.hfov_rad / 2.0),
                2.0 * altitude * np.tan(self.vfov_rad / 2.0))

    def gsd_m(self, altitude: float) -> float:
        """Ground sample distance [m/px] at nadir (horizontal axis)."""
        return altitude / self.fx


class GeolocationEngine:
    """
    Pixel -> ground-plane projection (flat earth, nadir camera, yaw-only).

        ray_cam   = K^-1 [u, v, 1]^T
        ray_body  = R_cb ray_cam          R_cb = diag(1, -1, -1)
        ray_world = R_bw(yaw) ray_body
        t         = -h / ray_world[z]
        (gx, gy)  = (x, y)_drone + t * ray_world[:2]

    Frame convention AS IMPLEMENTED (identical to pranav2.py):
        yaw = 0 -> East, CCW positive.  Image +u ("right") maps to the
        heading direction; image "up" maps 90 deg to the LEFT of heading.
    """

    def __init__(self, cam: CameraSpecs):
        self.cam = cam
        self.R_cb = np.array([[1.0, 0.0, 0.0],
                              [0.0, -1.0, 0.0],
                              [0.0, 0.0, -1.0]])

    def pixel_to_ground(self, u: float, v: float,
                        drone_x: float, drone_y: float,
                        altitude: float, yaw: float) -> Tuple[float, float]:
        ray_cam = self.cam.K_inv @ np.array([u, v, 1.0])
        ray_body = self.R_cb @ ray_cam
        c, s = np.cos(yaw), np.sin(yaw)
        R_bw = np.array([[c, -s, 0.0],
                         [s, c, 0.0],
                         [0.0, 0.0, 1.0]])
        ray_world = R_bw @ ray_body
        if abs(ray_world[2]) < 1e-12:
            return (float("nan"), float("nan"))
        t = -altitude / ray_world[2]
        return (drone_x + t * ray_world[0], drone_y + t * ray_world[1])

    def footprint_corners(self, drone_x: float, drone_y: float,
                          altitude: float, yaw: float) -> np.ndarray:
        px = [(0, 0), (self.cam.res_x, 0),
              (self.cam.res_x, self.cam.res_y), (0, self.cam.res_y)]
        return np.array([self.pixel_to_ground(u, v, drone_x, drone_y, altitude, yaw)
                         for u, v in px])


def geometry_selftest(eng: GeolocationEngine, alt: float) -> bool:
    """Cheap invariants that must hold before any detection is trusted."""
    cam = eng.cam
    ok = True

    # (1) principal point lands directly below the aircraft, any yaw
    for yaw in (0.0, 0.7, -2.1):
        gx, gy = eng.pixel_to_ground(cam.cx, cam.cy, 10.0, -5.0, alt, yaw)
        ok &= bool(np.hypot(gx - 10.0, gy + 5.0) < 1e-9)

    # (2) footprint edge lengths equal CameraSpecs.ground_footprint, any yaw
    w, h = cam.ground_footprint(alt)
    c = eng.footprint_corners(0.0, 0.0, alt, 0.7)   # TL, TR, BR, BL
    ok &= bool(abs(np.linalg.norm(c[1] - c[0]) - w) < 1e-6)
    ok &= bool(abs(np.linalg.norm(c[3] - c[0]) - h) < 1e-6)

    # (3) 1 px at the image centre == GSD
    g0 = np.array(eng.pixel_to_ground(cam.cx, cam.cy, 0, 0, alt, 0.0))
    g1 = np.array(eng.pixel_to_ground(cam.cx + 1, cam.cy, 0, 0, alt, 0.0))
    ok &= bool(abs(np.linalg.norm(g1 - g0) - cam.gsd_m(alt)) < 1e-9)
    return ok


# ════════════════════════════════════════════════════════════════════════
#  2.  DETECTION  (YOLOv8 via ultralytics, optional tiled inference)
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    @property
    def size_px(self) -> Tuple[float, float]:
        return (self.x2 - self.x1, self.y2 - self.y1)


def _tile_origins(length: int, tile: int, stride: int) -> List[int]:
    """Start offsets covering [0, length) with a flush final tile."""
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile, stride))
    origins.append(length - tile)
    return origins


class AerialPersonDetector:
    """Thin wrapper: COCO-pretrained YOLOv8, person class only."""

    def __init__(self, model_path: str = "yolov8s.pt",
                 conf: float = 0.10, imgsz: int = 640):
        try:
            from ultralytics import YOLO
        except ImportError:
            sys.exit("ultralytics not installed:  pip install ultralytics opencv-python")
        self.model = YOLO(model_path)   # auto-downloads weights if online
        self.conf = conf
        self.imgsz = imgsz

    def _infer(self, img_bgr: np.ndarray, imgsz: int,
               dx: int = 0, dy: int = 0) -> List[Detection]:
        res = self.model.predict(img_bgr, classes=[PERSON_CLASS_ID],
                                 conf=self.conf, imgsz=imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        return [Detection(b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy, float(c))
                for b, c in zip(xyxy, conf)]

    def detect(self, img_bgr: np.ndarray, tile: Optional[int] = None,
               overlap: float = 0.20) -> Tuple[List[Detection], int]:
        """Returns (detections sorted by confidence, number of forward passes)."""
        if tile is None:
            return sorted(self._infer(img_bgr, self.imgsz),
                          key=lambda d: -d.conf), 1

        if tile % 32:
            sys.exit("--tile must be a multiple of 32 (YOLO stride).")
        H, W = img_bgr.shape[:2]
        stride = max(1, int(tile * (1.0 - overlap)))
        dets: List[Detection] = []
        n_passes = 0
        for y0 in _tile_origins(H, tile, stride):
            for x0 in _tile_origins(W, tile, stride):
                crop = img_bgr[y0:y0 + tile, x0:x0 + tile]
                dets += self._infer(crop, tile, dx=x0, dy=y0)   # native resolution
                n_passes += 1
        return sorted(self._nms(dets), key=lambda d: -d.conf), n_passes

    @staticmethod
    def _nms(dets: List[Detection], iou_thr: float = 0.5) -> List[Detection]:
        """Merge duplicates produced by overlapping tiles."""
        if not dets:
            return dets
        import torch
        from torchvision.ops import nms   # installed as an ultralytics dependency
        boxes = torch.tensor([[d.x1, d.y1, d.x2, d.y2] for d in dets])
        scores = torch.tensor([d.conf for d in dets])
        return [dets[i] for i in nms(boxes, scores, iou_thr).tolist()]


# ════════════════════════════════════════════════════════════════════════
#  3.  ERROR BUDGET  (what does the (gx, gy) number actually mean?)
# ════════════════════════════════════════════════════════════════════════
def error_budget(cam: CameraSpecs, alt: float, offset_m: float) -> List[Tuple[str, float]]:
    """
    First-order 1-sigma ground error terms [m] for a target `offset_m` from
    nadir.  Input uncertainties are ASSUMED typical values for a small UAV
    with a consumer autopilot - not measured. Terms are combined by RSS.
    """
    gsd = cam.gsd_m(alt)
    terms = [
        ("bbox-centre jitter (2 px)",             2.0 * gsd),
        ("unmodelled roll/pitch (2 deg)",         alt * np.tan(np.radians(2.0))),
        ("heading error (5 deg)",                 offset_m * np.radians(5.0)),
        ("AGL error (3 m)",                       offset_m * 3.0 / alt),
        ("target height parallax (0.85 m)",       offset_m * 0.85 / (alt - 0.85)),
        ("GNSS position (2.5 m)",                 2.5),
    ]
    rss = float(np.sqrt(sum(t[1] ** 2 for t in terms)))
    return terms + [("TOTAL (RSS)", rss)]


# ════════════════════════════════════════════════════════════════════════
#  4.  REPORTING
# ════════════════════════════════════════════════════════════════════════
BAR = "=" * 72


def _wrap(text: str, indent: str = "   ") -> str:
    return textwrap.fill(text, width=72, initial_indent=indent,
                         subsequent_indent=indent + "  ")


def print_result(args, cam, det: Optional[Detection], gx, gy, n_passes, n_dets):
    w, h = cam.ground_footprint(args.alt)
    print(f"\n{BAR}\n   DETECTION + GEOLOCATION RESULT\n{BAR}")
    print(f"   Image            : {args.image}  ({cam.res_x}x{cam.res_y} px)")
    print(f"   Model            : {args.model}  imgsz={args.imgsz}  "
          f"conf>={args.conf}  tiling={'off' if not args.tile else args.tile}")
    print(f"   Forward passes   : {n_passes}   person candidates: {n_dets}")
    print(f"   Assumed pose     : drone=({args.x:.1f}, {args.y:.1f}) m, "
          f"AGL={args.alt:.0f} m, yaw={args.yaw_deg:.1f} deg "
          f"(0=East, CCW+){'  [image-top-is-nose: -90 deg applied]' if args.top_is_nose else ''}")
    print(f"   Footprint / GSD  : {w:.1f} x {h:.1f} m   |   GSD = {cam.gsd_m(args.alt)*100:.2f} cm/px")

    if det is None:
        print("\n   NO PERSON DETECTED at this confidence threshold.")
        print("   (Expected failure mode - see limitations below.)")
        return

    u, v = det.center
    bw, bh = det.size_px
    gsd = cam.gsd_m(args.alt)
    print(f"\n   Best detection   : conf = {det.conf:.2f}")
    print(f"   Bounding box     : ({det.x1:.0f},{det.y1:.0f})-({det.x2:.0f},{det.y2:.0f})  "
          f"= {bw:.0f} x {bh:.0f} px")
    print(f"   Centre pixel     : (u, v) = ({u:.1f}, {v:.1f})")
    print(f"   Ground position  : (gx, gy) = ({gx:.2f}, {gy:.2f}) m  [local frame]")

    # Scale-consistency check: is the bbox a plausible size for a person at this AGL?
    wm, hm = bw * gsd, bh * gsd
    print(f"   Implied box size : {wm:.2f} x {hm:.2f} m on the ground")
    if max(wm, hm) > 3.0 or max(wm, hm) < 0.2:
        print(_wrap("WARNING: implied size is not person-like. Either the photo was "
                    "not taken from the assumed altitude/camera, or this is a false "
                    "positive. The (gx, gy) number then demonstrates the projection "
                    "math only - it is NOT a validated position."))

    offset = float(np.hypot(gx - args.x, gy - args.y))
    print(f"\n   Error budget at {offset:.1f} m from nadir (assumed 1-sigma inputs):")
    for name, val in error_budget(cam, args.alt, offset):
        print(f"      {name:<36s} {val:6.2f} m")


def print_limitations(args, cam: CameraSpecs):
    gsd = cam.gsd_m(args.alt)
    person_px = 0.5 / gsd                         # ~0.5 m shoulder width seen from above
    scale = args.imgsz / max(cam.res_x, cam.res_y)
    person_px_resized = person_px * scale
    tile_scale = 1.0 if args.tile else scale
    print(f"\n{BAR}\n   TECHNICAL LIMITATIONS - COCO YOLO ON AERIAL IMAGERY\n{BAR}")
    notes = [
        f"SMALL-TARGET FLOOR. At {args.alt:.0f} m AGL this camera has GSD = "
        f"{gsd*100:.1f} cm/px, so a ~0.5 m wide person is ~{person_px:.0f} px across "
        f"at native resolution. YOLO resizes the full frame to {args.imgsz} px "
        f"(factor {scale:.2f}), shrinking that to ~{person_px_resized:.0f} px - far "
        f"below the ~16x16 px region where COCO-trained detectors degrade sharply. "
        f"The resize, not the sensor, is the main bottleneck.",

        "DOMAIN SHIFT. COCO 'person' is overwhelmingly side-on, large and "
        "well-lit. From nadir a person is a small blob of head/shoulders with "
        "foreshortened limbs and a long shadow. Expect low recall and poorly "
        "calibrated confidence. Fine-tune on VisDrone / HERIDAL / SARD "
        "(aerial, SAR-style) before trusting any score.",

        "FALSE POSITIVES. Rocks, bushes, fence posts, vehicle fragments and "
        "cast shadows resemble small human silhouettes. A low conf threshold "
        "(used here so a weak detection is visible) trades precision for "
        "recall - a real SAR pipeline needs temporal confirmation across "
        "overlapping frames (frontlap = 60 %) before alerting an operator.",

        "TILING / SAHI. Slicing the frame into native-resolution tiles "
        "(e.g. 640x640, 20 % overlap) keeps the target at its full "
        f"~{person_px:.0f} px size. Cost: a 4000x3000 frame needs 48 forward "
        "passes instead of 1; at 15 m/s with 60 % frontlap a frame arrives "
        "every ~2.2 s, i.e. ~22 inferences/s sustained - Jetson-class compute, "
        "not a Raspberry Pi. Tile borders also truncate boxes, hence "
        "cross-tile NMS. Run with --tile 640 to see it.",

        "GEOLOCATION CAVEATS. Flat-earth plane, nadir camera, yaw-only "
        "rotation. In a 35 deg banked turn the unmodelled roll displaces the "
        "footprint by tens of metres, so detections taken during turns "
        "should be gated out (or the model extended to full R = Rz Ry Rx "
        "with logged IMU attitude). Terrain relief and AGL errors scale the "
        "offset from nadir directly.",

        "VALIDATION STATUS. This demo uses one offline photo that was not "
        "captured by this airframe; it proves the pipeline wiring and the "
        "projection math (see self-test), not detection accuracy. "
        "Quantifying performance needs a labelled flight dataset and "
        "precision/recall at fixed IoU.",
    ]
    for i, n in enumerate(notes, 1):
        print(textwrap.fill(n, width=72, initial_indent=f"   {i}. ",
                            subsequent_indent="      "))
    print(BAR + "\n")


# ════════════════════════════════════════════════════════════════════════
#  5.  MAIN
# ════════════════════════════════════════════════════════════════════════
def annotate_and_save(img, det: Optional[Detection], gx, gy, out_path: Path):
    import cv2
    vis = img.copy()
    if det is not None:
        u, v = det.center
        cv2.rectangle(vis, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)),
                      (0, 255, 0), max(2, img.shape[1] // 800))
        cv2.drawMarker(vis, (int(u), int(v)), (0, 0, 255), cv2.MARKER_CROSS,
                       40, max(2, img.shape[1] // 800))
        label = f"person {det.conf:.2f} -> ({gx:.1f}, {gy:.1f}) m"
        scale = max(0.6, img.shape[1] / 1600)
        cv2.putText(vis, label, (int(det.x1), max(30, int(det.y1) - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default="aerial_sample.jpg")
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--alt", type=float, default=80.0, help="altitude AGL [m]")
    p.add_argument("--yaw-deg", type=float, default=0.0, help="heading [deg], 0=East, CCW+")
    p.add_argument("--x", type=float, default=0.0, help="drone local x [m]")
    p.add_argument("--y", type=float, default=0.0, help="drone local y [m]")
    p.add_argument("--conf", type=float, default=0.10, help="min confidence")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO input size (untiled mode)")
    p.add_argument("--tile", type=int, default=None, help="tile size px (multiple of 32)")
    p.add_argument("--top-is-nose", action="store_true",
                   help="camera mounted with image-top toward the nose "
                        "(applies yaw-90 deg to the model's image-right=heading convention)")
    p.add_argument("--out", default="aerial_sample_annotated.jpg")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        import cv2
    except ImportError:
        sys.exit("opencv not installed:  pip install opencv-python")

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot read '{args.image}'. Put an aerial photo next to this "
                 f"script or pass --image <path>.")
    H, W = img.shape[:2]

    # Intrinsics scaled to the actual image size (same sensor + lens, same FOV).
    cam = CameraSpecs(res_x=W, res_y=H)
    if abs((W / H) / (4000 / 3000) - 1.0) > 0.02:
        print(f"[warn] image aspect {W}/{H} != 4:3; assuming the photo is a "
              f"full-sensor frame. Cropped images change the true FOV.")
    eng = GeolocationEngine(cam)

    print(f"{BAR}\n   GEOMETRY SELF-TEST: "
          f"{'PASS' if geometry_selftest(eng, args.alt) else 'FAIL'}\n{BAR}")

    detector = AerialPersonDetector(args.model, conf=args.conf, imgsz=args.imgsz)
    dets, n_passes = detector.detect(img, tile=args.tile)
    best = dets[0] if dets else None

    gx = gy = float("nan")
    if best is not None:
        yaw = np.radians(args.yaw_deg - (90.0 if args.top_is_nose else 0.0))
        u, v = best.center
        gx, gy = eng.pixel_to_ground(u, v, args.x, args.y, args.alt, yaw)

    print_result(args, cam, best, gx, gy, n_passes, len(dets))
    print_limitations(args, cam)
    annotate_and_save(img, best, gx, gy, Path(args.out))
    print(f"Annotated image written to: {args.out}")


if __name__ == "__main__":
    main()
