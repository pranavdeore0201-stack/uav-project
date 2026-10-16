#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  FIXED-WING UAV GROUND CONTROL STATION SIMULATION                      ║
║  Autonomous Coverage Path Planning & Geolocation System                ║
║  ──────────────────────────────────────────────────────────────────────  ║
║  Author  : IvLabs Robotics Club Candidate                              ║
║  Airframe: Custom 1.4 m Fixed-Wing, Eppler E205, 1600 g AUW           ║
║  Modules : Boustrophedon CPP · Dubins Kinematics · Energy Model       ║
║            Pinhole Camera Geolocation · YOLO Detection Framework       ║
╚══════════════════════════════════════════════════════════════════════════╝

Dependencies:
    pip install numpy matplotlib

Optional (for Module 3 live inference — not required for simulation):
    pip install ultralytics opencv-python
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")           # Switch to "Agg" for headless / CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import Polygon as MplPolygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import textwrap, sys, time



# MODULE 0 ─ AIRFRAME & POWERTRAIN SPECIFICATION (HARDCODED)

@dataclass
class UAVSpecs:
    """
    Airframe + powertrain + 4S LiPo specification.
 
    Physics beyond a wings-level, constant-efficiency model
    -------------------------------------------------------
    (1) Load-factor penalty in banked turns.
        Coordinated, constant-altitude turn at bank angle phi:
            vertical :  L·cos(phi) = W        ->  n = L/W = 1/cos(phi)
            lateral  :  L·sin(phi) = m·V²/R   ->  R = V²/(g·tan(phi))
        Lift coefficient needed:  CL = 2·n·W / (rho·V²·S).
        With the parabolic polar CD = CD0 + k·CL², the induced drag is
            D_i = q·S·k·CL² = k·(n·W)² / (q·S) = n² · D_i(level)
        so induced drag scales with n² = 1/cos²(phi)  (+49 % at 35 deg).
        Parasite drag (q·S·CD0) is independent of n to first order.
 
    (2) Bus power = propulsive chain + constant avionics load:
            P_bus = P_aero / eta_total + P_avionics
 
    (3) Thevenin battery model (open-circuit voltage + series resistance):
            V_t = V_oc(SoC) - I·R_int
        An ESC is a constant-POWER load, so sag forces extra current; that
        extra current is dissipated as I²R heat inside the pack
        (see battery_current_A for the closed-form solution).
    """
    # ── Airframe geometry ──────────────────────────────────────────────
    wingspan_m: float          = 1.4
    root_chord_m: float        = 0.2067
    tip_chord_m: float         = 0.1033
    MAC_m: float               = 0.155
    taper_ratio: float         = 0.5
    wing_area_m2: float        = 0.217
    aspect_ratio: float        = 9.0
    airfoil: str               = "Eppler E205"
 
    # ── Weight & loading ───────────────────────────────────────────────
    AUW_kg: float              = 1.6
    CL_max: float              = 1.15           # @ Re ≈ 100 k, α ≈ 9°
    stall_speed_ms: float      = 10.4
 
    # ── Wing-level drag polar  CD = CD0 + k·CL² ───────────────────────
    CD0: float                 = 0.021
    k_induced: float           = 0.044
 
    # ── Propulsion & Avionics ──────────────────────────────────────────
    eta_total: float           = 0.55            # ESC(bus) -> aerodynamic power
    avionics_W: float          = 8.0             # Constant draw: Companion PC + FC + Payload
 
    # ── LiPo Electrochemistry (4S Pack) ────────────────────────────────
    battery_cells: int         = 4
    battery_V_nom: float       = 14.8            # Nominal voltage
    battery_mAh: float         = 5000.0          # Pack capacity
    battery_C_rate: int        = 20
    battery_V_full: float      = 16.8            # V_oc at SoC = 100% (4 x 4.20V)
    battery_V_empty: float     = 13.0            # V_oc at SoC = 0%   (4 x 3.25V)
    battery_R_int_ohm: float   = 0.040           # 40mΩ lumped internal resistance (I^2R losses)
    battery_V_cutoff: float    = 12.0            # Brown-out threshold
 
    # ── Atmosphere ─────────────────────────────────────────────────────
    rho: float                 = 1.225           # kg/m³  (ISA sea-level)
    g: float                   = 9.81            # m/s²
 
    # ── Flight envelope ────────────────────────────────────────────────
    cruise_speed_ms: float     = 15.0
    turn_speed_ms: float       = 12.0            # reduced for tighter turns
    max_bank_deg: float        = 35.0
    survey_altitude_m: float   = 80.0
 
    # Representative single-cell discharge curve (State of Charge vs Cell Voltage).
    # Un-annotated class attributes -> NOT dataclass fields.
    _OCV_SOC  = (0.00, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 1.00)
    _OCV_CELL = (3.25, 3.50, 3.62, 3.70, 3.76, 3.83, 3.98, 4.08, 4.20)
 
    def __post_init__(self):
        self.W_N          = self.AUW_kg * self.g
        self.bank_rad     = np.radians(self.max_bank_deg)
        self.R_min_cruise = self.cruise_speed_ms**2 / (self.g * np.tan(self.bank_rad))
        self.R_min_turn   = self.turn_speed_ms**2   / (self.g * np.tan(self.bank_rad))
 
        # Pack charge in coulombs:  1 mAh = 1e-3 A · 3600 s = 3.6 C
        self.battery_Q_C  = self.battery_mAh * 3.6
        # Continuous current ceiling from the C-rating (20C · 5 Ah = 100 A)
        self.I_max_A      = self.battery_C_rate * self.battery_mAh / 1000.0
 
        # Pack OCV curve: normalise the single-cell curve to [0, 1] and
        # stretch it onto [V_empty, V_full] so the endpoints match the spec
        # exactly (13.0 V at SoC 0, 16.8 V at SoC 1) while keeping the
        # characteristic LiPo knee at low SoC and the flat plateau mid-range.
        soc   = np.array(self._OCV_SOC)
        cell  = np.array(self._OCV_CELL)
        shape = (cell - cell[0]) / (cell[-1] - cell[0])
        self._ocv_soc  = soc
        self._ocv_pack = self.battery_V_empty + (self.battery_V_full - self.battery_V_empty) * shape
        self.battery_Wh = self.battery_V_nom * (self.battery_mAh / 1000.0)
 
    # ── Aerodynamics with load-factor penalty ──────────────────────────
    def load_factor(self, bank_rad: float = 0.0) -> float:
        """
        n = L/W = 1/cos(phi) for a coordinated level turn.
        Clamped at 80 deg to avoid the 1/cos singularity at 90 deg.
        """
        return 1.0 / np.cos(min(abs(bank_rad), np.radians(80.0)))
 
    def CL_for_level_flight(self, V: float, bank_rad: float = 0.0) -> float:
        """
        Lift coefficient required to hold altitude:
            n·W = q·S·CL   ->   CL = 2·n·W / (rho·V²·S)
        """
        n = self.load_factor(bank_rad)
        return (2.0 * n * self.W_N) / (self.rho * V**2 * self.wing_area_m2)
 
    def CD_from_polar(self, CL: float) -> float:
        return self.CD0 + self.k_induced * CL**2
 
    def parasite_drag_N(self, V: float) -> float:
        """D_0 = q·S·CD0   (independent of load factor to first order)."""
        return 0.5 * self.rho * V**2 * self.wing_area_m2 * self.CD0
 
    def induced_drag_N(self, V: float, bank_rad: float = 0.0) -> float:
        """
        D_i = q·S·k·CL²  with  CL = n·W/(q·S)
            = k·(n·W)² / (q·S)
            = n² · D_i(level)
        The quadratic dependence on lift is why a 35 deg bank (n = 1.22)
        costs +49 % induced drag, not +22 %.
        """
        CL = self.CL_for_level_flight(V, bank_rad)
        q  = 0.5 * self.rho * V**2
        return q * self.wing_area_m2 * self.k_induced * CL**2
 
    def drag_N(self, V: float, bank_rad: float = 0.0) -> float:
        return self.parasite_drag_N(V) + self.induced_drag_N(V, bank_rad)
 
    def power_aero_W(self, V: float, bank_rad: float = 0.0) -> float:
        """Power required (thrust power) at steady speed:  P = D·V."""
        return self.drag_N(V, bank_rad) * V
 
    def stall_speed_banked_ms(self, bank_rad: float = 0.0) -> float:
        """
        Stall speed rises with the square root of load factor:
            V_s(phi) = V_s · sqrt(n)     (CL_max fixed, L = n·W)
        """
        return self.stall_speed_ms * np.sqrt(self.load_factor(bank_rad))
 
    # ── Electrical bus & battery ───────────────────────────────────────
    def power_bus_W(self, V: float, bank_rad: float = 0.0) -> float:
        """
        Total electrical power drawn from the ESC input bus:
            P_bus = P_aero / eta_total + P_avionics
        eta_total lumps ESC + motor + propeller efficiency.
        """
        return (self.power_aero_W(V, bank_rad) / self.eta_total) + self.avionics_W
 
    def power_elec_W(self, V: float, bank_rad: float = 0.0) -> float:
        """
        Bus power (kept under this name for the GCS telemetry panel).
        Single definition: includes avionics and, if given, bank penalty.
        """
        return self.power_bus_W(V, bank_rad)
 
    def ocv_V(self, soc: float) -> float:
        """Open-circuit pack voltage from State of Charge via the OCV table."""
        return float(np.interp(np.clip(soc, 0.0, 1.0), self._ocv_soc, self._ocv_pack))
 
    def battery_current_A(self, P_bus: float, V_oc: float) -> Tuple[float, bool]:
        """
        Pack current required to deliver P_bus to a constant-power load.
 
        Derivation
        ----------
            V_t   = V_oc - I·R                  (Thevenin source)
            P_bus = V_t · I = V_oc·I - R·I²
            =>  R·I² - V_oc·I + P_bus = 0
            =>  I = [V_oc ± sqrt(V_oc² - 4·R·P_bus)] / (2R)
 
        The '+' root has V_t < V_oc/2 (unstable, non-physical operating
        point); the '-' root is the physical one. Rationalising it
        (multiply by [V_oc + sqrt]/[V_oc + sqrt]) gives
            I = 2·P_bus / (V_oc + sqrt(V_oc² - 4·R·P_bus))
        which is numerically stable as R -> 0 (reduces to P/V_oc) and has
        no catastrophic cancellation.
 
        Feasibility: the radicand is >= 0 only if P_bus <= V_oc²/(4R)
        (maximum-power-transfer limit). Past it the pack cannot deliver the
        demanded power -> returns (V_oc/(2R), True) and the caller must
        treat it as a brown-out.
 
        Energy bookkeeping: V_oc·I = V_t·I + I²R = P_bus + P_heat.
        """
        R = self.battery_R_int_ohm
        disc = V_oc**2 - 4.0 * R * P_bus
        if disc < 0.0:
            return V_oc / (2.0 * R), True
        return 2.0 * P_bus / (V_oc + np.sqrt(disc)), False
 
    # ── Characteristic speeds (parabolic polar) ────────────────────────
    def optimal_endurance_speed(self) -> float:
        """V for min aerodynamic power (max endurance).  CL* = √(3·CD0 / k)"""
        CL_star = np.sqrt(3.0 * self.CD0 / self.k_induced)
        return np.sqrt(2.0 * self.W_N / (self.rho * CL_star * self.wing_area_m2))
 
    def optimal_range_speed(self) -> float:
        """V for max L/D (max range).  CL* = √(CD0 / k)"""
        CL_star = np.sqrt(self.CD0 / self.k_induced)
        return np.sqrt(2.0 * self.W_N / (self.rho * CL_star * self.wing_area_m2))

# MODULE 0-B ─ PINHOLE CAMERA SPECIFICATION

@dataclass
class CameraSpecs:
    """
    Simulated nadir-pointing survey camera.
    Sensor dimensions and focal length define the pinhole model.
    """
    sensor_w_mm: float = 6.17          # 1/2.3″ sensor width
    sensor_h_mm: float = 4.55          # 1/2.3″ sensor height
    focal_mm: float    = 4.5           # focal length
    res_x: int         = 4000          # horizontal pixels
    res_y: int         = 3000          # vertical pixels

    def __post_init__(self):
        # Focal length in pixels
        self.fx = self.focal_mm * self.res_x / self.sensor_w_mm
        self.fy = self.focal_mm * self.res_y / self.sensor_h_mm
        self.cx = self.res_x / 2.0
        self.cy = self.res_y / 2.0
        # Intrinsic matrix K  (3×3)
        self.K = np.array([
            [self.fx, 0,       self.cx],
            [0,       self.fy, self.cy],
            [0,       0,       1      ]
        ])
        self.K_inv = np.linalg.inv(self.K)
        # Angular FOV
        self.hfov_rad = 2.0 * np.arctan(self.sensor_w_mm / (2.0 * self.focal_mm))
        self.vfov_rad = 2.0 * np.arctan(self.sensor_h_mm / (2.0 * self.focal_mm))

    def ground_footprint(self, altitude: float) -> Tuple[float, float]:
        """Returns (width_m, height_m) of the camera footprint on the ground."""
        w = 2.0 * altitude * np.tan(self.hfov_rad / 2.0)
        h = 2.0 * altitude * np.tan(self.vfov_rad / 2.0)
        return w, h


# MODULE 1-A ─ DUBINS PATH SOLVER  (pure geometry, no black boxes)

def _mod2pi(angle: float) -> float:
    """Map angle to [0, 2π)."""
    return angle % (2.0 * np.pi)


def _dubins_LSL(alpha: float, beta: float, d: float):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    p_sq = 2 + d*d - 2*np.cos(alpha - beta) + 2*d*(sa - sb)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    theta = np.arctan2(cb - ca, d + sa - sb)
    t = _mod2pi(-alpha + theta)
    q = _mod2pi(beta - theta)
    return t, p, q, "LSL"


def _dubins_RSR(alpha: float, beta: float, d: float):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    p_sq = 2 + d*d - 2*np.cos(alpha - beta) + 2*d*(sb - sa)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    theta = np.arctan2(ca - cb, d - sa + sb)
    t = _mod2pi(alpha - theta)
    q = _mod2pi(-beta + theta)
    return t, p, q, "RSR"


def _dubins_LSR(alpha: float, beta: float, d: float):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    p_sq = -2 + d*d + 2*np.cos(alpha - beta) + 2*d*(sa + sb)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    theta = np.arctan2(-ca - cb, d + sa + sb) - np.arctan2(-2.0, p)
    t = _mod2pi(-alpha + theta)
    q = _mod2pi(-_mod2pi(beta) + theta)
    return t, p, q, "LSR"


def _dubins_RSL(alpha: float, beta: float, d: float):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    p_sq = -2 + d*d + 2*np.cos(alpha - beta) - 2*d*(sa + sb)
    if p_sq < 0:
        return None
    p = np.sqrt(p_sq)
    theta = np.arctan2(ca + cb, d - sa - sb) - np.arctan2(2.0, p)
    t = _mod2pi(alpha - theta)
    q = _mod2pi(_mod2pi(beta) - theta)
    return t, p, q, "RSL"


def _dubins_RLR(alpha: float, beta: float, d: float):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    tmp = (6.0 - d*d + 2*np.cos(alpha - beta) + 2*d*(sa - sb)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2*np.pi - np.arccos(tmp))
    theta = np.arctan2(ca - cb, d - sa + sb)
    t = _mod2pi(alpha - theta + _mod2pi(p / 2.0))
    q = _mod2pi(alpha - beta - t + _mod2pi(p))
    return t, p, q, "RLR"


def _dubins_LRL(alpha: float, beta: float, d: float):
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta),  np.sin(beta)
    tmp = (6.0 - d*d + 2*np.cos(alpha - beta) + 2*d*(sb - sa)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = _mod2pi(2*np.pi - np.arccos(tmp))
    theta = np.arctan2(-ca + cb, d + sa - sb)
    t = _mod2pi(-alpha + theta + _mod2pi(p / 2.0))
    q = _mod2pi(_mod2pi(beta) - alpha - t + _mod2pi(p))
    return t, p, q, "LRL"


def dubins_shortest_path(
    start: Tuple[float, float, float],
    end:   Tuple[float, float, float],
    radius: float
) -> Optional[Dict]:
    """
    Solve for the shortest Dubins path between two configurations.

    Parameters
    ----------
    start, end : (x, y, theta)  —  position (m) and heading (rad)
    radius     : minimum turn radius (m)

    Returns
    -------
    dict with keys:
        'type'     : str  e.g. "LSR"
        'segments' : (t, p, q) normalised arc / straight lengths
        'length'   : total path length in metres
        'radius'   : turn radius used
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    D  = np.hypot(dx, dy)
    d  = D / radius                              # normalised distance

    if D < 1e-10:
        # Degenerate — start ≈ end in position
        d = 0.0
        phi = 0.0
    else:
        phi = np.arctan2(dy, dx)                 # bearing start → end

    alpha = _mod2pi(start[2] - phi)
    beta  = _mod2pi(end[2]   - phi)

    solvers = [_dubins_LSL, _dubins_RSR, _dubins_LSR,
               _dubins_RSL, _dubins_RLR, _dubins_LRL]
    best, best_cost = None, float('inf')
    for solver in solvers:
        result = solver(alpha, beta, d)
        if result is not None:
            t, p, q, name = result
            cost = t + p + q
            if cost < best_cost:
                best_cost = cost
                best = {'type': name, 'segments': (t, p, q),
                        'length': cost * radius, 'radius': radius}
    return best


def sample_dubins_path(
    start: Tuple[float, float, float],
    path_info: Dict,
    step: float = 1.0
) -> np.ndarray:
    """
    Sample (x, y, theta) points along a solved Dubins path at spacing `step` m.

    Each letter in path_info['type'] maps to the corresponding normalised
    segment length in path_info['segments'].  L/R are arcs; S is straight.

    Returns
    -------
    np.ndarray of shape (N, 3)  —  columns are (x, y, theta).
    """
    r = path_info['radius']
    seg_types = list(path_info['type'])           # e.g. ['L', 'S', 'R']
    seg_params = list(path_info['segments'])      # normalised (t, p, q)

    x, y, theta = start
    pts = []

    for stype, param in zip(seg_types, seg_params):
        if stype == 'S':
            # ── Straight segment ── length = param * r
            length = param * r
            n = max(int(length / step), 1)
            for i in range(n):
                frac = i / n
                pts.append([x + frac * length * np.cos(theta),
                            y + frac * length * np.sin(theta),
                            theta])
            x += length * np.cos(theta)
            y += length * np.sin(theta)

        elif stype == 'L':
            # ── Left (CCW) arc ── arc angle = param
            cx = x - r * np.sin(theta)
            cy = y + r * np.cos(theta)
            start_ang = np.arctan2(y - cy, x - cx)
            arc_len = param * r
            n = max(int(arc_len / step), 1)
            for i in range(n):
                frac = i / n
                a = start_ang + frac * param
                pts.append([cx + r * np.cos(a),
                            cy + r * np.sin(a),
                            theta + frac * param])
            theta += param
            x = cx + r * np.cos(start_ang + param)
            y = cy + r * np.sin(start_ang + param)

        elif stype == 'R':
            # ── Right (CW) arc ── arc angle = param
            cx = x + r * np.sin(theta)
            cy = y - r * np.cos(theta)
            start_ang = np.arctan2(y - cy, x - cx)
            arc_len = param * r
            n = max(int(arc_len / step), 1)
            for i in range(n):
                frac = i / n
                a = start_ang - frac * param
                pts.append([cx + r * np.cos(a),
                            cy + r * np.sin(a),
                            theta - frac * param])
            theta -= param
            x = cx + r * np.cos(start_ang - param)
            y = cy + r * np.sin(start_ang - param)

    pts.append([x, y, theta])
    return np.array(pts)


# MODULE 1-B ─ BOUSTROPHEDON COVERAGE PATH PLANNER

class CoveragePathPlanner:
    """
    Generates a Boustrophedon (lawn-mower) path over a rectangular
    survey polygon, connecting parallel sweeps with Dubins U-turns.

    Swath width is computed dynamically from camera FOV, altitude,
    and desired sidelap fraction.
    """

    def __init__(self, uav: UAVSpecs, cam: CameraSpecs,
                 sidelap: float = 0.20, frontlap: float = 0.60):
        self.uav      = uav
        self.cam      = cam
        self.sidelap  = sidelap
        self.frontlap = frontlap

        # ── Derive swath geometry ──────────────────────────────────────
        fp_w, fp_h = cam.ground_footprint(uav.survey_altitude_m)
        self.footprint_w = fp_w                # full cross-track width
        self.footprint_h = fp_h                # full along-track height
        self.swath_w = fp_w * (1.0 - sidelap)  # effective cross-track
        self.spacing_along = fp_h * (1.0 - frontlap)  # inter-frame

    # ── public API ─────────────────────────────────────────────────────

    def plan(self, survey_rect: Tuple[float, float, float, float],
             heading_deg: float = 0.0
    ) -> Tuple[np.ndarray, List[Dict], List[np.ndarray]]:
        """
        Parameters
        ----------
        survey_rect : (x_min, y_min, x_max, y_max) in metres (local frame)
        heading_deg : sweep heading in degrees (0 = East)

        Returns
        -------
        waypoints   : (N, 3) array  —  every sampled (x, y, theta) on path
        legs        : list of dicts describing each leg type & speed
        sweep_lines : list of 2-point arrays for each parallel line
        """
        x0, y0, x1, y1 = survey_rect
        sweep_heading = np.radians(heading_deg)

        # ── Lay down parallel sweep lines ─────────────────────────────
        n_lines = int(np.ceil((y1 - y0) / self.swath_w)) + 1
        sweep_lines = []
        for i in range(n_lines):
            y_line = y0 + i * self.swath_w
            if y_line > y1 + self.swath_w * 0.5:
                break
            if i % 2 == 0:
                # East-bound
                sweep_lines.append(np.array([
                    [x0, y_line, sweep_heading],
                    [x1, y_line, sweep_heading]
                ]))
            else:
                # West-bound (reversed heading)
                sweep_lines.append(np.array([
                    [x1, y_line, sweep_heading + np.pi],
                    [x0, y_line, sweep_heading + np.pi]
                ]))

        # ── Connect sweeps with Dubins turns ──────────────────────────
        all_pts = []
        legs    = []
        r_turn  = self.uav.R_min_turn

        for idx, line in enumerate(sweep_lines):
            # Straight sweep
            start_cfg = tuple(line[0])
            end_cfg   = tuple(line[1])
            seg_len   = np.hypot(end_cfg[0] - start_cfg[0],
                                 end_cfg[1] - start_cfg[1])
            n_pts = max(int(seg_len / 2.0), 2)
            xs = np.linspace(start_cfg[0], end_cfg[0], n_pts)
            ys = np.linspace(start_cfg[1], end_cfg[1], n_pts)
            th = np.full(n_pts, start_cfg[2])
            sweep_pts = np.column_stack([xs, ys, th])
            all_pts.append(sweep_pts)
            legs.append({'type': 'sweep', 'speed': self.uav.cruise_speed_ms,
                         'length': seg_len, 'points': len(sweep_pts)})

            # Dubins turn to next sweep (except after last)
            if idx < len(sweep_lines) - 1:
                turn_start = tuple(line[1])
                turn_end   = tuple(sweep_lines[idx + 1][0])
                path_info  = dubins_shortest_path(turn_start, turn_end, r_turn)
                if path_info is None:
                    # Fallback: straight connector (should not happen)
                    cxs = np.linspace(turn_start[0], turn_end[0], 20)
                    cys = np.linspace(turn_start[1], turn_end[1], 20)
                    cth = np.full(20, turn_end[2])
                    turn_pts = np.column_stack([cxs, cys, cth])
                else:
                    turn_pts = sample_dubins_path(turn_start, path_info, step=1.5)
                all_pts.append(turn_pts)
                legs.append({'type': 'turn', 'speed': self.uav.turn_speed_ms,
                             'length': path_info['length'] if path_info else 0,
                             'points': len(turn_pts)})

        waypoints = np.vstack(all_pts)
        return waypoints, legs, sweep_lines


# MODULE 2 ─ ENDURANCE-AWARE ENERGY MODEL

class EnergyModel:
    """
    Integrates electrical power consumption along the flight path using an
    active battery model: SoC-dependent open-circuit voltage, I²R sag,
    time-stepped Coulomb counting and Joule-heat accumulation.
 
    Per-leg bank angle: 'turn' legs fly at the UAV's max bank angle (load
    factor n = 1/cos(phi), induced drag x n²); 'sweep' legs are wings-level.
    NOTE: the whole turn leg is treated as banked, which is conservative —
    the straight (S) segment inside a Dubins CSC path is really wings-level.
    """
    def __init__(self, uav: UAVSpecs):
        self.uav = uav
 
    def analyse_mission(self, waypoints: np.ndarray, legs: List[Dict], dt: float = 0.1) -> Dict:
        u = self.uav
        R = u.battery_R_int_ohm
        Q = u.battery_Q_C                      # pack charge [C]
 
        total_chem_energy_J = 0.0              # ∫ V_oc·I dt   (energy leaving the cells)
        total_heat_energy_J = 0.0              # ∫ I²R dt      (lost inside the pack)
        total_time_s        = 0.0
        leg_reports         = []
        wp_idx              = 0
 
        soc        = 1.0
        v_term_min = np.inf
        i_peak     = 0.0
        brownout   = False                     # P_bus exceeded V_oc²/(4R) at any step
        over_curr  = False                     # I exceeded C-rating at any step
 
        for leg in legs:
            V     = leg['speed']
            n_pts = leg['points']
 
            # Banked only on turn legs -> n = 1/cos(phi), D_i x n²
            bank_angle = u.bank_rad if leg['type'] == 'turn' else 0.0
 
            # Along-track distance of this leg from the sampled waypoints
            seg_pts = waypoints[wp_idx: wp_idx + n_pts]
            if len(seg_pts) > 1:
                diffs = np.diff(seg_pts[:, :2], axis=0)
                dist  = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
            else:
                dist = leg['length']
            t_seg = dist / V if V > 0 else 0.0
 
            # Bus power is constant within a leg (V and phi are constant);
            # the CURRENT is not, because V_oc falls as SoC depletes.
            P_bus = u.power_bus_W(V, bank_rad=bank_angle)
 
            # ── Time-stepped integration inside the leg ───────────────
            n_steps = max(1, int(np.ceil(t_seg / dt)))
            dt_i    = t_seg / n_steps          # equal sub-steps, sum to t_seg exactly
 
            leg_E_J = leg_heat_J = leg_C = 0.0
            leg_vt_min, leg_i_peak = np.inf, 0.0
 
            for _ in range(n_steps):
                V_oc = u.ocv_V(soc)                                   # EMF at present SoC
                I, sat = u.battery_current_A(P_bus, V_oc)             # quadratic solution
                V_t    = V_oc - I * R                                 # sagged terminal voltage
                P_heat = I * I * R                                    # Joule heating  I²R
 
                # V_oc·I = V_t·I + I²R = P_bus + P_heat
                leg_E_J    += V_oc * I * dt_i
                leg_heat_J += P_heat * dt_i
                leg_C      += I * dt_i
 
                # Coulomb counting:  SoC <- SoC - I·dt / Q
                soc = max(0.0, soc - I * dt_i / Q)
 
                leg_vt_min = min(leg_vt_min, V_t)
                leg_i_peak = max(leg_i_peak, I)
                brownout  |= sat
                over_curr |= (I > u.I_max_A)
 
            total_chem_energy_J += leg_E_J
            total_heat_energy_J += leg_heat_J
            total_time_s        += t_seg
            v_term_min = min(v_term_min, leg_vt_min)
            i_peak     = max(i_peak, leg_i_peak)
 
            leg_reports.append({
                'type':      leg['type'],
                'speed':     V,
                'dist_m':    dist,
                'time_s':    t_seg,
                'power_W':   (leg_E_J / t_seg) if t_seg > 0 else P_bus,   # mean chemical power
                'energy_Wh': leg_E_J / 3600.0,
                'voltage':   leg_vt_min,                                  # lowest V_t in leg
                'current_A': (leg_C / t_seg) if t_seg > 0 else leg_i_peak,  # mean current
                # ── extras (ignored by existing UI) ──
                'bank_deg':   float(np.degrees(bank_angle)),
                'CL':         float(u.CL_for_level_flight(V, bank_angle)),
                'power_bus_W': P_bus,
                'heat_Wh':    leg_heat_J / 3600.0,
                'soc_end':    soc,
            })
            wp_idx += n_pts
 
        used_Wh    = total_chem_energy_J / 3600.0
        heat_Wh    = total_heat_energy_J / 3600.0
        margin_pct = soc * 100.0
        if not np.isfinite(v_term_min):
            v_term_min = u.battery_V_full
 
        return {
            'total_energy_Wh': used_Wh,
            'heat_loss_Wh':    heat_Wh,
            'battery_Wh':      u.battery_Wh,
            'flight_time_s':   total_time_s,
            'margin_pct':      margin_pct,
            'v_term_min':      v_term_min,
            'i_peak_A':        i_peak,
            'go':              (margin_pct > 20.0)
                               and (v_term_min > u.battery_V_cutoff)
                               and (not brownout) and (not over_curr),
            'legs':            leg_reports,
            # ── extras (ignored by existing UI) ──
            'bus_energy_Wh':   used_Wh - heat_Wh,     # energy actually delivered to the bus
            'brownout':        brownout,
            'over_current':    over_curr,
        }
 
    def optimal_sweep_speed(self) -> Tuple[float, float]:
        """Speed minimising energy per metre, P_bus(V)/V, avionics included."""
        V_range = np.linspace(self.uav.stall_speed_ms * 1.1, 25.0, 500)
        E_per_m = np.array([self.uav.power_bus_W(V) / V for V in V_range])
        idx = np.argmin(E_per_m)
        return float(V_range[idx]), float(self.uav.power_bus_W(V_range[idx]))
 
    def print_preflight_report(self, mission: Dict, planner, survey_rect: Tuple):
        u = self.uav
        x0, y0, x1, y1 = survey_rect
        area_m2 = (x1 - x0) * (y1 - y0)
 
        V_opt_sweep, P_opt = self.optimal_sweep_speed()
        V_endurance = u.optimal_endurance_speed()
 
        sweep_legs = [l for l in mission['legs'] if l['type'] == 'sweep']
        turn_legs  = [l for l in mission['legs'] if l['type'] == 'turn']
        total_sweep_dist = sum(l['dist_m'] for l in sweep_legs)
        total_turn_dist  = sum(l['dist_m'] for l in turn_legs)
 
        n_turn   = u.load_factor(u.bank_rad)
        CL_turn  = u.CL_for_level_flight(u.turn_speed_ms, u.bank_rad)
        Vs_bank  = u.stall_speed_banked_ms(u.bank_rad)
        stall_mg = u.turn_speed_ms / Vs_bank
 
        hdr = "=" * 70
        print(f"\n{hdr}")
        print("   PRE-FLIGHT MISSION REPORT  —  ADVANCED THERMAL/AERO MODEL")
        print(hdr)
        print(f"   Airframe        : {u.airfoil},  b = {u.wingspan_m} m,  S = {u.wing_area_m2} m²")
        print(f"   AUW             : {u.AUW_kg:.2f} kg  ({u.W_N:.1f} N)")
        print(f"   Drag polar      : CD = {u.CD0} + {u.k_induced}·CL²")
        print(f"   Battery         : {u.battery_cells}S {u.battery_mAh:.0f}mAh ({u.battery_Wh:.1f} Wh), R_int = {u.battery_R_int_ohm*1000:.0f}mΩ")
        print(f"   Total efficiency: η = {u.eta_total*100:.0f} % (Excluding I²R losses)")
        print(f"   Avionics Draw   : {u.avionics_W:.1f} W")
        print(f"{hdr}")
        print(f"   Survey area     : {(x1-x0):.0f} × {(y1-y0):.0f} m  ({area_m2/1e4:.2f} ha)")
        print(f"   Altitude AGL    : {u.survey_altitude_m:.0f} m")
        print(f"   Effective swath : {planner.swath_w:.1f} m")
        print(f"   Min turn radius : {u.R_min_turn:.1f} m  (@ {u.turn_speed_ms} m/s, {u.max_bank_deg}° bank)")
        print(f"{hdr}")
        print(f"   V_stall         : {u.stall_speed_ms:.1f} m/s  (wings level)")
        print(f"   V_stall @ bank  : {Vs_bank:.1f} m/s  (x√n)")
        print(f"   V_endurance     : {V_endurance:.1f} m/s  (min aero power)")
        print(f"   V_optimal sweep : {V_opt_sweep:.1f} m/s  (min E/m incl. avionics)")
        print(f"   Turn load factor: {n_turn:.2f}g (Induced drag penalty: +{((n_turn**2)-1)*100:.0f}%)")
        print(f"   Turn CL         : {CL_turn:.2f}  (CL_max = {u.CL_max:.2f})")
        print(f"   Turn stall mgn  : V/Vs = {stall_mg:.2f}" +
              ("   ⚠ below 1.20 — consider higher turn speed or lower bank" if stall_mg < 1.2 else ""))
        print(f"{hdr}")
        print(f"   Sweep distance  : {total_sweep_dist:.0f} m")
        print(f"   Turn distance   : {total_turn_dist:.0f} m")
        print(f"   Flight time     : {mission['flight_time_s']:.0f} s  ({mission['flight_time_s']/60:.1f} min)")
        print(f"   Chem Energy Req : {mission['total_energy_Wh']:.2f} Wh")
        print(f"   Thermal Loss    : {mission['heat_loss_Wh']:.2f} Wh (Joule heating)")
        print(f"   Terminal V_min  : {mission['v_term_min']:.2f} V")
        print(f"   Peak Current    : {mission['i_peak_A']:.2f} A")
        print(f"   Remaining Margin: {mission['margin_pct']:.1f} % (State of Charge)")
        verdict = "✓  GO" if mission['go'] else "✗  NO-GO  (Limits Exceeded)"
        print(f"   VERDICT         : {verdict}")
        print(f"{hdr}\n")


# MODULE 3 ─ PERSON DETECTION FRAMEWORK (YOLO PLACEHOLDER)

class PersonDetector:
    """
    Placeholder / mock module demonstrating aerial person detection
    with a pretrained YOLO model on VisDrone or HERIDAL-type imagery.

    Running this in 'mock' mode (default) generates synthetic detections
    and prints an honest assessment of expected performance.

    To run LIVE inference set `live=True` and ensure `ultralytics` is
    installed:  pip install ultralytics
    """

    # ── Expected performance on aerial datasets (literature survey) ───
    MOCK_METRICS = {
        'dataset': 'VisDrone-DET val (person class)',
        'model': 'YOLOv8s  (pretrained COCO → fine-tuned VisDrone)',
        'input_res': 640,
        'precision': 0.58,
        'recall': 0.42,
        'mAP_50': 0.44,
        'mAP_50_95': 0.24,
        'notes': [
            'Small-target AP drops sharply below 16×16 px bounding box.',
            'At 80 m AGL a standing person subtends ~6–10 px height '
            'with the specified camera → near or below YOLO receptive-'
            'field floor; expect Recall < 0.3 without tiling / SAHI.',
            'False-positive rate rises on cluttered terrain '
            '(vehicles, shadows, rocks) — expect precision ~0.5.',
            'Windy conditions cause motion blur during mapping passes '
            'at 15 m/s; effective resolution degrades further.',
            'Thermal / NIR sensor fusion dramatically improves recall '
            'for SAR missions but is outside this airframe budget.',
        ]
    }

    def __init__(self, live: bool = False, model_path: str = 'yolov8s.pt'):
        self.live = live
        self.model_path = model_path
        self.model = None
        if live:
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
            except ImportError:
                print("[DetectionModule] ultralytics not installed — "
                      "falling back to mock mode.")
                self.live = False

    def detect(self, image_or_path) -> List[Dict]:
        """Run detection.  Returns list of {bbox, conf, class}."""
        if self.live and self.model is not None:
            results = self.model(image_or_path, verbose=False)
            dets = []
            for r in results:
                for box in r.boxes:
                    dets.append({
                        'bbox': box.xyxy[0].tolist(),
                        'conf': float(box.conf),
                        'cls':  int(box.cls),
                    })
            return dets
        # ── Mock detection ─────────────────────────────────────────────
        return [{'bbox': [120, 80, 145, 130], 'conf': 0.72, 'cls': 0}]

    def print_classification_report(self):
        m = self.MOCK_METRICS
        hdr = "=" * 70
        print(f"\n{hdr}")
        print("   AERIAL PERSON DETECTION — EXPECTED PERFORMANCE REPORT")
        print(hdr)
        print(f"   Model       : {m['model']}")
        print(f"   Dataset     : {m['dataset']}")
        print(f"   Input size  : {m['input_res']}×{m['input_res']} px")
        print(f"   Precision   : {m['precision']:.2f}")
        print(f"   Recall      : {m['recall']:.2f}")
        print(f"   mAP@0.50    : {m['mAP_50']:.2f}")
        print(f"   mAP@0.50:95 : {m['mAP_50_95']:.2f}")
        print(f"{hdr}")
        print("   HONEST ASSESSMENT OF LIMITATIONS:")
        for i, note in enumerate(m['notes'], 1):
            wrapped = textwrap.fill(note, width=60,
                                   initial_indent=f"   {i}. ",
                                   subsequent_indent="      ")
            print(wrapped)
        print(f"{hdr}\n")


# MODULE 4 ─ PINHOLE CAMERA GEOLOCATION MODEL

class GeolocationEngine:
    """
    Projects a pixel coordinate from an aerial image onto the ground
    plane using the classic pinhole camera model.

    Derivation
    ----------
    1.  Homogeneous pixel → normalised image ray:
            ray_cam = K⁻¹ · [u, v, 1]ᵀ

    2.  Camera-to-body rotation (nadir-pointing, z-down):
            R_cb = [[1,0,0],[0,-1,0],[0,0,-1]]
        so  ray_body = R_cb · ray_cam

    3.  Body-to-world rotation (yaw only, no roll / pitch for a
        nadir survey camera on a trimmed aircraft):
            R_bw = [[cos ψ, -sin ψ, 0],
                    [sin ψ,  cos ψ, 0],
                    [  0,      0,   1]]
        so  ray_world = R_bw · ray_body

    4.  Intersect with ground plane z = 0  (drone at altitude h):
            t = -h / ray_world[2]
            ground_xy = drone_xy + t * ray_world[:2]
    """

    def __init__(self, cam: CameraSpecs):
        self.cam = cam
        # Camera → Body rotation (nadir: optical axis = body -Z)
        self.R_cb = np.array([[1,  0,  0],
                              [0, -1,  0],
                              [0,  0, -1]], dtype=float)

    def pixel_to_ground(
        self,
        u: float, v: float,
        drone_x: float, drone_y: float,
        altitude: float, yaw: float
    ) -> Tuple[float, float]:
        """
        Parameters
        ----------
        u, v     : pixel coordinates (col, row)
        drone_x, drone_y : drone position in local frame (m)
        altitude : AGL (m)
        yaw      : heading (rad), 0 = East, CCW positive

        Returns
        -------
        (gx, gy) : ground coordinates in local frame (m)
        """
        # Step 1 — pixel → camera ray
        ray_cam = self.cam.K_inv @ np.array([u, v, 1.0])

        # Step 2 — camera → body
        ray_body = self.R_cb @ ray_cam

        # Step 3 — body → world  (yaw-only rotation about Z)
        c, s = np.cos(yaw), np.sin(yaw)
        R_bw = np.array([[ c, -s, 0],
                         [ s,  c, 0],
                         [ 0,  0, 1]], dtype=float)
        ray_world = R_bw @ ray_body

        # Step 4 — intersect with z = 0
        if abs(ray_world[2]) < 1e-12:
            return (np.nan, np.nan)       # ray parallel to ground
        t = altitude / ray_world[2]       # note: ray_world[2] < 0 for nadir
        # We want the point *below*, so positive t when ray is downward:
        t = -altitude / ray_world[2]      # corrected sign

        gx = drone_x + t * ray_world[0]
        gy = drone_y + t * ray_world[1]
        return (gx, gy)

    def footprint_corners(
        self,
        drone_x: float, drone_y: float,
        altitude: float, yaw: float
    ) -> np.ndarray:
        """
        Returns the 4 ground-plane corners of the camera footprint
        as a (4, 2) array  [TL, TR, BR, BL]  suitable for Polygon patches.
        """
        corners_px = [
            (0,              0),              # top-left
            (self.cam.res_x, 0),              # top-right
            (self.cam.res_x, self.cam.res_y), # bottom-right
            (0,              self.cam.res_y),  # bottom-left
        ]
        ground = []
        for u, v in corners_px:
            gx, gy = self.pixel_to_ground(u, v, drone_x, drone_y,
                                          altitude, yaw)
            ground.append([gx, gy])
        return np.array(ground)


# MODULE 5 ─ GCS ANIMATION  (10 Hz Matplotlib)

from stl import mesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches

class GCSAnimation:
    """
    High-fidelity 3D Ground Control Station display with live telemetry,
    moving sensor footprint, battery bar, CAD model rotation, and detection overlay.
    """
    def __init__(self, waypoints: np.ndarray, legs: List[Dict],
                 sweep_lines: List[np.ndarray],
                 survey_rect: Tuple[float, float, float, float],
                 uav: UAVSpecs, cam: CameraSpecs,
                 mission: Dict,
                 person_ground: Tuple[float, float] = None):
        self.wp    = waypoints
        self.legs  = legs
        self.lines = sweep_lines
        self.rect  = survey_rect
        self.uav   = uav
        self.cam   = cam
        self.mission = mission
        self.geo   = GeolocationEngine(cam)
        self.person_ground = person_ground
        self.person_detected = False

        self.n_wp = len(waypoints)
        self.cum_dist = np.zeros(self.n_wp)
        for i in range(1, self.n_wp):
            self.cum_dist[i] = self.cum_dist[i-1] + np.hypot(
                waypoints[i, 0] - waypoints[i-1, 0],
                waypoints[i, 1] - waypoints[i-1, 1])

        self.wp_speed = np.zeros(self.n_wp)
        idx = 0
        for leg in legs:
            speed = leg['speed']
            for _ in range(leg['points']):
                if idx < self.n_wp:
                    self.wp_speed[idx] = speed
                    idx += 1

        self.wp_time = np.zeros(self.n_wp)
        for i in range(1, self.n_wp):
            d = self.cum_dist[i] - self.cum_dist[i-1]
            v = self.wp_speed[i] if self.wp_speed[i] > 0 else 1.0
            self.wp_time[i] = self.wp_time[i-1] + d / v

        self.total_time = self.wp_time[-1]

    def build(self):
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(16, 10), facecolor='#0a0e17')
        self.fig.canvas.manager.set_window_title('IvLabs 3D GCS — Fixed-Wing CPP Simulation')

        # 3D Main map
        self.ax = self.fig.add_axes([0.05, 0.05, 0.72, 0.90], projection='3d')
        self.ax.set_facecolor('#0f1520')
        x0, y0, x1, y1 = self.rect
        self.ax.set_xlim(x0 - 80, x1 + 80)
        self.ax.set_ylim(y0 - 80, y1 + 80)
        self.ax.set_zlim(0, self.uav.survey_altitude_m + 30)
        
        self.ax.set_xlabel('East (m)', color='#88aacc', fontsize=10)
        self.ax.set_ylabel('North (m)', color='#88aacc', fontsize=10)
        self.ax.set_zlabel('Altitude AGL (m)', color='#88aacc', fontsize=10)
        self.ax.tick_params(colors='#556677')

        # Draw the Ground Plane at Z=0
        X, Y = np.meshgrid([x0, x1], [y0, y1])
        self.ax.plot_surface(X, Y, np.zeros_like(X), color='#00cc88', alpha=0.1)

        # Planned path (faint 3D line at altitude)
        self.ax.plot(self.wp[:, 0], self.wp[:, 1], zs=self.uav.survey_altitude_m, color='#223344', lw=1.0, zorder=1)

        # Covered track (updated each frame)
        self.track_line, = self.ax.plot([], [], zs=[], color='#00ffaa', lw=1.5, zorder=2)

        # Camera footprint polygon for 3D axis
        self.footprint_patch = Poly3DCollection([np.zeros((4, 3))], facecolors='#ffaa0020', edgecolors='#ffaa00', lw=0.8)
        self.ax.add_collection3d(self.footprint_patch)

        # Person target mapping on 3D plane
        if self.person_ground is not None:
            px, py = self.person_ground
            self.person_marker, = self.ax.plot([px], [py], zs=[0], marker='*', color='#ff3366', markersize=14, alpha=0.0)
        else:
            self.person_marker = None

        # Load STL or fallback to 3D marker
        try:
            self.uav_mesh = mesh.Mesh.from_file('uav_model.stl')
            self.base_vectors = self.uav_mesh.vectors.copy() 
            self.drone_poly = Poly3DCollection(self.base_vectors, facecolors='#ff6633', edgecolors='k', alpha=0.9)
            self.ax.add_collection3d(self.drone_poly)
            self.using_3d_model = True
        except FileNotFoundError:
            self.drone_marker, = self.ax.plot([], [], [], marker='^', color='#ff6633', markersize=10)
            self.using_3d_model = False

        # ── Telemetry panel (Static 2D overlay on right side) ──
        self.ax_telem = self.fig.add_axes([0.80, 0.05, 0.18, 0.90])
        self.ax_telem.set_facecolor('#0a0e17')
        self.ax_telem.axis('off')

        self.telem_texts = {}
        # Professional MFD (Multi-Function Display) Labels
        labels = [
            ('STATUS',   '[ SYS ] MISSION'),
            ('TIME',     '[ TIM ] FLIGHT TIME'),
            ('SPEED',    '[ SPD ] AIRSPEED'),
            ('HEADING',  '[ NAV ] HEADING'),
            ('ALTITUDE', '[ ALT ] ALT AGL'),
            ('DISTANCE', '[ DST ] DISTANCE'),
            ('BATTERY',  '[ BAT ] PACK CHARGE'),
            ('POWER',    '[ PWR ] BUS POWER'),
            ('LEG',      '[ MOD ] FLIGHT MODE'),
            ('DETECTED', '[ TGT ] TARGETING')
        ]
        
        y_positions = np.linspace(0.92, 0.12, len(labels))
        for (key, display_str), yp in zip(labels, y_positions):
            # Left aligned label (Dimmer, technical look)
            self.ax_telem.text(0.05, yp, display_str, fontsize=9, fontfamily='monospace', color='#557799', transform=self.ax_telem.transAxes, verticalalignment='center')
            # Right aligned value (Brighter, Bold, Dynamic)
            self.telem_texts[key] = self.ax_telem.text(0.95, yp, '', fontsize=11, fontfamily='monospace', color='#ffffff', transform=self.ax_telem.transAxes, verticalalignment='center', horizontalalignment='right', fontweight='bold')

        # Battery bar background
        self.batt_bg = mpatches.FancyBboxPatch((0.05, 0.02), 0.90, 0.04, boxstyle="round,pad=0.005", facecolor='#1a1a2e', edgecolor='#334455', transform=self.ax_telem.transAxes, clip_on=False)
        self.ax_telem.add_patch(self.batt_bg)
        
        self.batt_bar = mpatches.FancyBboxPatch((0.05, 0.02), 0.90, 0.04, boxstyle="round,pad=0.005", facecolor='#00cc66', edgecolor='none', transform=self.ax_telem.transAxes, clip_on=False)
        self.ax_telem.add_patch(self.batt_bar)

        self.fig.text(0.40, 0.97, 'IvLabs 3D GCS — Fixed-Wing CPP Mission', fontsize=14, fontfamily='monospace', color='#00ffaa', ha='center', va='top', fontweight='bold')

    def _get_wp_index(self, t: float) -> int:
        idx = np.searchsorted(self.wp_time, t, side='right') - 1
        return max(0, min(idx, self.n_wp - 1))

    def _point_in_polygon(self, px, py, poly: np.ndarray) -> bool:
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi):
                inside = not inside
            j = i
        return inside

    def update(self, frame: int):
        try:
            t   = min(frame * 0.1, self.total_time)
            idx = self._get_wp_index(t)
            x, y, theta = self.wp[idx]
            alt = self.uav.survey_altitude_m
 
            # ── Scalars consumed by the telemetry panel (all defined here) ──
            elapsed     = t                                    # mission clock [s]
            heading_deg = float(np.degrees(theta) % 360.0)     # same convention as planner: 0 = East, CCW
            speed       = float(self.wp_speed[idx])
            dist        = float(self.cum_dist[idx])
 
            # Flight mode of the current waypoint (sweep / turn)
            cum_pts = 0
            leg_type = "—"
            for leg in self.legs:
                cum_pts += leg['points']
                if idx < cum_pts:
                    leg_type = leg['type'].upper()
                    break
 
            # Bank angle: turns fly at max bank (load factor n = 1/cos(phi)); sweeps are wings-level
            current_bank = self.uav.bank_rad if leg_type == 'TURN' else 0.0
 
            # Bus power incl. avionics and the n² induced-drag penalty
            pwr = self.uav.power_bus_W(speed, bank_rad=current_bank)
 
            # ── Battery state of charge (built once, then O(1) per frame) ──
            # Same Thevenin model as EnergyModel.analyse_mission, stepped per waypoint, so the
            # on-screen % tracks the pre-flight report instead of a separate nominal-Wh estimate.
            if getattr(self, '_soc_trace', None) is None:
                u = self.uav
                bank = np.zeros(self.n_wp)
                k = 0
                for leg in self.legs:
                    for _ in range(leg['points']):
                        if k < self.n_wp:
                            bank[k] = u.bank_rad if leg['type'] == 'turn' else 0.0
                            k += 1
                soc = 1.0
                trace = np.ones(self.n_wp)
                for i in range(1, self.n_wp):
                    dt_i = self.wp_time[i] - self.wp_time[i - 1]
                    P_i  = u.power_bus_W(self.wp_speed[i], bank_rad=bank[i])
                    I_i, _ = u.battery_current_A(P_i, u.ocv_V(soc))
                    soc = max(0.0, soc - I_i * dt_i / u.battery_Q_C)     # Coulomb counting
                    trace[i] = soc
                self._soc_trace = trace
            batt_pct = 100.0 * float(self._soc_trace[idx])
 
            # ── Update 3D track line ──
            self.track_line.set_data(self.wp[:idx+1, 0], self.wp[:idx+1, 1])
            self.track_line.set_3d_properties(np.full(idx+1, alt))
 
            # ── Rotate and position the 3D drone model ──
            if self.using_3d_model:
                c, s = np.cos(theta), np.sin(theta)
                R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                transformed = np.dot(self.base_vectors, R_z.T) + [x, y, alt]
                self.drone_poly.set_verts(transformed)
            else:
                self.drone_marker.set_data([x], [y])
                self.drone_marker.set_3d_properties([alt])
 
            # ── Camera footprint on the ground plane (Z = 0) ──
            corners_2d = self.geo.footprint_corners(x, y, alt, theta)
            corners_3d = np.column_stack((corners_2d, np.zeros(4)))
            self.footprint_patch.set_verts([corners_3d])
 
            # ── Target detection logic ──
            if self.person_ground is not None and not self.person_detected:
                if self._point_in_polygon(self.person_ground[0], self.person_ground[1], corners_2d):
                    self.person_detected = True
                    if self.person_marker is not None:
                        self.person_marker.set_alpha(1.0)
 
            # ── UI text & colour updates ──
            self.telem_texts['STATUS'].set_text('ACTIVE' if t < self.total_time else 'COMPLETE')
            self.telem_texts['STATUS'].set_color('#00ff88' if t < self.total_time else '#ffaa00')
 
            self.telem_texts['TIME'].set_text(f'{elapsed:.1f} s')
            self.telem_texts['SPEED'].set_text(f'{speed:.1f} m/s')
            self.telem_texts['HEADING'].set_text(f'{heading_deg:03.0f}°')
            self.telem_texts['ALTITUDE'].set_text(f'{alt:.0f} m')
            self.telem_texts['DISTANCE'].set_text(f'{dist:.0f} m')
            self.telem_texts['BATTERY'].set_text(f'{batt_pct:.1f} %')
 
            # Power turns red only while banked, i.e. when the n² induced-drag penalty is active.
            # (A fixed 30 W threshold is always exceeded now that the 8 W avionics load is included.)
            self.telem_texts['POWER'].set_text(f'{pwr:.1f} W')
            self.telem_texts['POWER'].set_color('#ff4444' if current_bank > 0.0 else '#00ffaa')
 
            self.telem_texts['LEG'].set_text(leg_type)
            self.telem_texts['LEG'].set_color('#00cc88' if leg_type == 'SWEEP' else '#ffaa00')
 
            det_str = 'LOCKED' if self.person_detected else 'SCANNING...'
            self.telem_texts['DETECTED'].set_text(det_str)
            self.telem_texts['DETECTED'].set_color('#ff3366' if self.person_detected else '#557799')
 
            # ── Dynamic battery bar ──
            frac = max(batt_pct / 100.0, 0.0)
            self.batt_bar.set_width(0.90 * frac)
            if batt_pct > 50:   self.batt_bar.set_facecolor('#00cc66')
            elif batt_pct > 20: self.batt_bar.set_facecolor('#ffaa00')
            else:               self.batt_bar.set_facecolor('#ff3333')
 
        except Exception:
            # Matplotlib swallows exceptions raised inside timer callbacks, which looks like a
            # silent freeze. Print the full stack trace, then stop the timer so it prints once.
            import traceback
            print(f"\n[GCSAnimation.update] frame {frame} FAILED:", file=sys.stderr)
            traceback.print_exc()
            sys.stderr.flush()
            anim = getattr(self, 'anim', None)
            if anim is not None:
                try:
                    anim.event_source.stop()
                except Exception:
                    pass
 
        return (self.track_line,)
 

    def run(self, speedup: float = 8.0):
        self.build()
        n_frames = int(self.total_time / 0.1) + 10
        interval_ms = int(100.0 / speedup) 
        self.anim = animation.FuncAnimation(self.fig, self.update, frames=n_frames, interval=interval_ms, blit=False, repeat=False)
        plt.show()

# MAIN — ORCHESTRATOR

def main():
    print("\n" + "─" * 70)
    print("  Initialising IvLabs GCS Simulation …")
    print("─" * 70)

    # ── 1. Instantiate specifications ─────────────────────────────────
    uav = UAVSpecs()
    cam = CameraSpecs()

    print(f"  Airframe : {uav.airfoil}, b = {uav.wingspan_m} m, "
          f"AUW = {uav.AUW_kg} kg")
    print(f"  Min turn radius (cruise) : {uav.R_min_cruise:.1f} m  "
          f"@ {uav.cruise_speed_ms} m/s")
    print(f"  Min turn radius (turn)   : {uav.R_min_turn:.1f} m  "
          f"@ {uav.turn_speed_ms} m/s")

    # ── 2. Define survey area ─────────────────────────────────────────
    survey_rect = (0.0, 0.0, 600.0, 400.0)      # 600 × 400 m
    print(f"  Survey area : {survey_rect[2]:.0f} × {survey_rect[3]:.0f} m")

    # ── 3. Plan coverage path ─────────────────────────────────────────
    planner = CoveragePathPlanner(uav, cam, sidelap=0.20, frontlap=0.60)
    print(f"  Camera footprint : {planner.footprint_w:.1f} × "
          f"{planner.footprint_h:.1f} m")
    print(f"  Effective swath  : {planner.swath_w:.1f} m")

    waypoints, legs, sweep_lines = planner.plan(survey_rect, heading_deg=0)
    print(f"  Waypoints generated : {len(waypoints)}")
    print(f"  Legs (sweep + turn) : {len(legs)}")

    # ── 4. Energy analysis ────────────────────────────────────────────
    energy = EnergyModel(uav)
    mission = energy.analyse_mission(waypoints, legs)
    energy.print_preflight_report(mission, planner, survey_rect)

    # ── 5. Detection framework report ─────────────────────────────────
    detector = PersonDetector(live=False)
    detector.print_classification_report()

    # ── 6. Mock geolocation demo ──────────────────────────────────────
    geo = GeolocationEngine(cam)
    print("=" * 70)
    print("   GEOLOCATION DEMO  —  Pinhole Camera Projection")
    print("=" * 70)
    # Suppose YOLO detected a person at pixel (2200, 1800) while the
    # drone was at (300, 200) heading East (0 rad) at 80 m AGL.
    demo_u, demo_v = 2200, 1800
    demo_drone = (300.0, 200.0, 80.0, 0.0)
    gx, gy = geo.pixel_to_ground(demo_u, demo_v,
                                  demo_drone[0], demo_drone[1],
                                  demo_drone[2], demo_drone[3])
    print(f"   Drone position  : ({demo_drone[0]}, {demo_drone[1]}) m,  "
          f"alt = {demo_drone[2]} m,  ψ = {np.degrees(demo_drone[3]):.0f}°")
    print(f"   Pixel detection : ({demo_u}, {demo_v})")
    print(f"   Ground estimate : ({gx:.1f}, {gy:.1f}) m")

    # Place a synthetic "person" at a location the flight path will cover
    person_pos = (300.0, planner.swath_w * 2.5)
    print(f"   Simulated person: ({person_pos[0]:.0f}, {person_pos[1]:.0f}) m")
    print("=" * 70)

    # ── 7. Launch animation ───────────────────────────────────────────
    print("\n  Launching GCS animation …  (close window to exit)\n")
    gcs = GCSAnimation(waypoints, legs, sweep_lines,
                        survey_rect, uav, cam, mission,
                        person_ground=person_pos)
    gcs.run(speedup=10.0)

    print("\n  Simulation complete.\n")


if __name__ == '__main__':
    main()