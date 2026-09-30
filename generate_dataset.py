"""
generate_dataset.py
--------------------
Synthetic-but-realistic dataset generator for the "AI in Smart City Planning"
project (building compliance + underground infrastructure conflict).

Why synthetic: Google's paid GIS/Maps APIs are expensive for a prototype.
This generator encodes real Indian building-bylaw LOGIC (FAR/FSI slabs tied
to road width, height-vs-road-width rules, setback tables, parking norms,
ground coverage caps) drawn from common Indian Development Control
Regulations (DCR/UDCPR-style) and the National Building Code (NBC 2016)
structure, then samples proposals and labels them exactly the way a real
scrutiny system would. Random noise + "as-built deviation" + "underground
utility conflict" fields are layered on top to mimic real-world messiness
(the same gaps called out in the PPT: outdated GIS, approval-vs-actual
mismatch, incomplete utility records).

Swap-in path to real data later (see README.md) — this script's schema is
built so you can replace generate_synthetic_rows() with rows pulled from
Bhuvan/OSM/state GIS portals without touching the training code.
"""

import numpy as np
import pandas as pd
import random
import argparse
from pathlib import Path

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------------------
# 1. Reference tables encoding simplified Indian DCR-style norms
# ---------------------------------------------------------------------------

CITIES = ["Bhopal", "Indore", "Pune", "Mumbai", "Delhi", "Bengaluru",
          "Hyderabad", "Ahmedabad", "Jaipur", "Nagpur"]

ZONES = ["Residential", "Commercial", "Mixed-Use", "Industrial", "Institutional"]

# FAR/FSI slabs keyed by road width (metres) -> base permissible FAR
def permissible_far(road_width, zone):
    if road_width < 9:
        base = 1.0
    elif road_width < 12:
        base = 1.5
    elif road_width < 18:
        base = 2.0
    elif road_width < 24:
        base = 2.5
    else:
        base = 3.0
    zone_bonus = {"Residential": 0.0, "Commercial": 0.5, "Mixed-Use": 0.3,
                  "Industrial": -0.5, "Institutional": 0.2}
    return max(0.5, base + zone_bonus[zone])

# Height permissible (m) tied to road width + zone cap (simplified NBC-style)
def permissible_height(road_width, zone):
    raw = 1.5 * road_width + 6
    cap = {"Residential": 45, "Commercial": 60, "Mixed-Use": 50,
           "Industrial": 30, "Institutional": 40}[zone]
    return min(raw, cap)

# Setback norms (m) as a function of building height (typical DCR stepped table)
def required_setbacks(height):
    if height <= 10:
        return {"front": 3.0, "rear": 3.0, "side": 1.5}
    elif height <= 15:
        return {"front": 4.5, "rear": 3.0, "side": 2.0}
    elif height <= 24:
        return {"front": 6.0, "rear": 4.5, "side": 3.0}
    elif height <= 36:
        return {"front": 7.5, "rear": 6.0, "side": 4.0}
    else:
        return {"front": 9.0, "rear": 7.5, "side": 5.0}

# Ground coverage cap (%) by zone
GROUND_COVERAGE_CAP = {"Residential": 65, "Commercial": 60,
                        "Mixed-Use": 60, "Industrial": 50, "Institutional": 55}

# Parking norm: ECS (equivalent car spaces) required per 100 sqm built-up area
PARKING_NORM = {"Residential": 1.0, "Commercial": 2.0, "Mixed-Use": 1.5,
                "Industrial": 0.5, "Institutional": 1.0}

# Typical underground utility depth bands (m) by utility type
UTILITY_DEPTH = {
    "water_main": (0.9, 1.8),
    "sewerage": (1.2, 3.0),
    "storm_drain": (1.0, 2.5),
    "electric_cable": (0.6, 1.2),
    "telecom_duct": (0.5, 1.0),
}

# ---------------------------------------------------------------------------
# 2. Row generator
# ---------------------------------------------------------------------------

def sample_row(idx):
    city = random.choice(CITIES)
    zone = random.choices(ZONES, weights=[0.4, 0.2, 0.2, 0.1, 0.1])[0]
    plot_size = round(np.random.uniform(80, 2200), 1)          # sq.m
    road_width = round(np.random.choice(
        [4.5, 6, 7.5, 9, 12, 15, 18, 24, 30],
        p=[0.05, 0.15, 0.15, 0.2, 0.15, 0.12, 0.08, 0.06, 0.04]), 1)

    far_permissible = permissible_far(road_width, zone)
    height_permissible = permissible_height(road_width, zone)

    # Draw "intended" values with a tendency to hover near the limit
    # (real proposals cluster near what's allowed) plus a violation
    # probability layer.
    violate_far = np.random.rand() < 0.28
    if violate_far:
        far_proposed = round(far_permissible * np.random.uniform(1.03, 1.35), 2)
    else:
        far_proposed = round(far_permissible * np.random.uniform(0.55, 0.99), 2)

    violate_height = np.random.rand() < 0.22
    if violate_height:
        height_proposed = round(height_permissible * np.random.uniform(1.02, 1.25), 1)
    else:
        height_proposed = round(height_permissible * np.random.uniform(0.5, 0.98), 1)

    req_setback = required_setbacks(height_proposed)
    violate_setback = np.random.rand() < 0.30
    setback_front = round(req_setback["front"] * (np.random.uniform(0.55, 0.9)
                           if violate_setback and np.random.rand() < 0.5
                           else np.random.uniform(1.0, 1.4)), 2)
    setback_rear = round(req_setback["rear"] * (np.random.uniform(0.55, 0.9)
                          if violate_setback and np.random.rand() < 0.5
                          else np.random.uniform(1.0, 1.4)), 2)
    setback_side = round(req_setback["side"] * (np.random.uniform(0.55, 0.9)
                          if violate_setback and np.random.rand() < 0.5
                          else np.random.uniform(1.0, 1.4)), 2)

    coverage_cap = GROUND_COVERAGE_CAP[zone]
    violate_coverage = np.random.rand() < 0.20
    ground_coverage_pct = round(coverage_cap * (np.random.uniform(1.02, 1.2)
                                 if violate_coverage else np.random.uniform(0.5, 0.99)), 1)

    footprint = round(plot_size * (ground_coverage_pct / 100), 1)
    floors = max(1, round(height_proposed / 3.2))
    built_up_area = round(footprint * floors, 1)

    parking_required_ecs = round((built_up_area / 100) * PARKING_NORM[zone], 1)
    violate_parking = np.random.rand() < 0.25
    parking_provided_ecs = round(parking_required_ecs * (np.random.uniform(0.4, 0.9)
                                  if violate_parking else np.random.uniform(1.0, 1.3)), 1)

    # --- ground truth compliance labels (computed from the rules above) ---
    far_violation = int(far_proposed > far_permissible)
    height_violation = int(height_proposed > height_permissible)
    setback_violation = int(setback_front < req_setback["front"] or
                             setback_rear < req_setback["rear"] or
                             setback_side < req_setback["side"])
    coverage_violation = int(ground_coverage_pct > coverage_cap)
    parking_violation = int(parking_provided_ecs < parking_required_ecs)
    overall_violation = int(any([far_violation, height_violation,
                                  setback_violation, coverage_violation,
                                  parking_violation]))

    # --- approved-vs-actual construction deviation (independent event) ---
    deviation_occurred = int(np.random.rand() < 0.18)
    deviation_type = "none"
    if deviation_occurred:
        deviation_type = random.choice(
            ["extra_floor_added", "height_increase", "setback_encroachment",
             "footprint_expansion", "unauthorized_basement"])

    # --- underground utility conflict simulation ---
    # Designed so BOTH visible features (foundation depth, GIS record
    # confidence, basement, road width) and a hidden ground-truth utility
    # layout drive the outcome -- i.e. genuinely learnable from visible
    # data, but not perfectly, which is realistic (this is exactly why the
    # PPT calls for a probability/confidence score rather than a hard yes/no).
    basement_proposed = int(np.random.rand() < 0.35)
    foundation_depth = round(np.random.uniform(1.0, 6.0)
                              if basement_proposed else np.random.uniform(0.8, 2.0), 2)
    utility_record_confidence = round(np.random.uniform(0.55, 0.98), 2)  # GIS record quality
    utility_present = random.choices(
        list(UTILITY_DEPTH.keys()) + ["none"],
        weights=[0.18, 0.15, 0.12, 0.12, 0.1, 0.33])[0]
    utility_depth_est = None
    if utility_present != "none":
        lo, hi = UTILITY_DEPTH[utility_present]
        utility_depth_est = round(np.random.uniform(lo, hi), 2)

    # visible-feature risk score (what a model without ground truth CAN learn)
    depth_component = min(foundation_depth / 6.0, 1.0) * 0.55
    confidence_component = (1 - utility_record_confidence) * 0.35
    basement_component = 0.10 * basement_proposed
    road_component = 0.10 if road_width < 9 else 0.0  # narrow-road utilities are shallower/denser
    visible_risk = depth_component + confidence_component + basement_component + road_component

    # hidden ground-truth component (irreducible uncertainty for the model)
    hidden_risk = 0.0
    if utility_present != "none":
        buffer_m = 0.5
        hidden_risk = 0.35 if foundation_depth > (utility_depth_est - buffer_m) else 0.05

    conflict_prob = min(0.95, max(0.02, 0.5 * visible_risk + hidden_risk))
    utility_conflict = int(np.random.rand() < conflict_prob)

    # --- XAI-style explanation string (rule-grounded, not a black box output) ---
    reasons = []
    if far_violation:
        reasons.append(f"Proposed FAR {far_proposed} exceeds permissible FAR {far_permissible}.")
    if height_violation:
        reasons.append(f"Proposed height {height_proposed} m exceeds permissible height {round(height_permissible,1)} m.")
    if setback_violation:
        if setback_front < req_setback["front"]:
            reasons.append(f"Front setback {setback_front} m < required {req_setback['front']} m.")
        if setback_rear < req_setback["rear"]:
            reasons.append(f"Rear setback {setback_rear} m < required {req_setback['rear']} m.")
        if setback_side < req_setback["side"]:
            reasons.append(f"Side setback {setback_side} m < required {req_setback['side']} m.")
    if coverage_violation:
        reasons.append(f"Ground coverage {ground_coverage_pct}% exceeds cap {coverage_cap}%.")
    if parking_violation:
        reasons.append(f"Parking provided {parking_provided_ecs} ECS < required {parking_required_ecs} ECS.")
    if utility_conflict:
        if utility_present != "none":
            reasons.append(f"Foundation depth {foundation_depth} m conflicts with estimated {utility_present} at {utility_depth_est} m (confidence {utility_record_confidence}).")
        else:
            reasons.append(f"Foundation depth {foundation_depth} m has elevated utility-conflict risk because GIS record confidence is {utility_record_confidence}.")
    if not reasons:
        reasons.append("No violations detected against applicable norms.")
    explanation = " ".join(reasons)

    return {
        "proposal_id": f"PROP-{idx:06d}",
        "city": city,
        "zone_type": zone,
        "plot_size_sqm": plot_size,
        "road_width_m": road_width,
        "far_proposed": far_proposed,
        "far_permissible": far_permissible,
        "height_proposed_m": height_proposed,
        "height_permissible_m": round(height_permissible, 1),
        "setback_front_m": setback_front,
        "setback_rear_m": setback_rear,
        "setback_side_m": setback_side,
        "ground_coverage_pct": ground_coverage_pct,
        "footprint_sqm": footprint,
        "floors": floors,
        "built_up_area_sqm": built_up_area,
        "parking_required_ecs": parking_required_ecs,
        "parking_provided_ecs": parking_provided_ecs,
        "basement_proposed": basement_proposed,
        "foundation_depth_m": foundation_depth,
        "nearby_utility_type": utility_present,
        "utility_est_depth_m": utility_depth_est,
        "gis_record_confidence": utility_record_confidence,
        "far_violation": far_violation,
        "height_violation": height_violation,
        "setback_violation": setback_violation,
        "coverage_violation": coverage_violation,
        "parking_violation": parking_violation,
        "utility_conflict": utility_conflict,
        "overall_violation": overall_violation,
        "post_approval_deviation": deviation_occurred,
        "deviation_type": deviation_type,
        "explanation": explanation,
    }


def generate_synthetic_rows(n_rows: int) -> pd.DataFrame:
    rows = [sample_row(i) for i in range(n_rows)]
    return pd.DataFrame(rows)


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate synthetic smart-city compliance data.")
    parser.add_argument("--rows", type=int, default=12000,
                        help="Number of rows to generate (default: 12000).")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "india_smart_city_compliance_dataset.csv",
        help="CSV output path.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.rows < 1:
        raise ValueError("--rows must be at least 1")

    df = generate_synthetic_rows(args.rows)
    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.head(3).to_string())
    print("\nViolation rate columns:")
    for c in ["far_violation", "height_violation", "setback_violation",
              "coverage_violation", "parking_violation", "utility_conflict",
              "overall_violation", "post_approval_deviation"]:
        print(f"  {c}: {df[c].mean():.3f}")
