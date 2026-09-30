"""Interactive inference helpers for smart-city building proposals."""

from pathlib import Path

import joblib
import pandas as pd

from generate_dataset import (
    GROUND_COVERAGE_CAP,
    PARKING_NORM,
    permissible_far,
    permissible_height,
    required_setbacks,
)

PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

VIOLATION_LABELS = [
    "far_violation",
    "height_violation",
    "setback_violation",
    "coverage_violation",
    "parking_violation",
]

FEATURES_A = [
    "plot_size_sqm", "road_width_m", "far_proposed", "far_permissible",
    "height_proposed_m", "height_permissible_m", "setback_front_m",
    "setback_rear_m", "setback_side_m", "ground_coverage_pct",
    "footprint_sqm", "floors", "built_up_area_sqm", "parking_required_ecs",
    "parking_provided_ecs", "city", "zone_type",
]

FEATURES_B = [
    "plot_size_sqm", "road_width_m", "foundation_depth_m",
    "basement_proposed", "gis_record_confidence", "floors",
    "built_up_area_sqm", "city", "zone_type",
]


def load_models(artifacts_dir=ARTIFACTS_DIR):
    """Load the trained models without retraining them."""
    compliance_path = Path(artifacts_dir) / "model_a_compliance.joblib"
    utility_path = Path(artifacts_dir) / "model_b_utility_conflict.joblib"
    if not compliance_path.exists() or not utility_path.exists():
        raise FileNotFoundError(
            "Trained model artifacts are missing. Run train_model.py first."
        )
    return joblib.load(compliance_path), joblib.load(utility_path)


def build_proposal(inputs):
    """Derive regulatory fields from the fields entered by the user."""
    city = inputs["city"]
    zone_type = inputs["zone_type"]
    plot_size = float(inputs["plot_size_sqm"])
    road_width = float(inputs["road_width_m"])
    floors = int(inputs["floors"])
    ground_coverage = float(inputs["ground_coverage_pct"])
    built_up_area = round(plot_size * ground_coverage / 100 * floors, 1)
    parking_required = round(
        built_up_area / 100 * PARKING_NORM[zone_type], 1
    )

    return {
        **inputs,
        "plot_size_sqm": plot_size,
        "road_width_m": road_width,
        "floors": floors,
        "ground_coverage_pct": ground_coverage,
        "far_permissible": permissible_far(road_width, zone_type),
        "height_permissible_m": round(permissible_height(road_width, zone_type), 1),
        "footprint_sqm": round(plot_size * ground_coverage / 100, 1),
        "built_up_area_sqm": built_up_area,
        "parking_required_ecs": parking_required,
        "basement_proposed": int(bool(inputs["basement_proposed"])),
    }


def check_feasibility(inputs, models=None):
    """Return model predictions, risk, rule checks, and a user-facing decision."""
    proposal = build_proposal(inputs)
    model_a, model_b = models or load_models()
    row_a = pd.DataFrame([{key: proposal[key] for key in FEATURES_A}])
    row_b = pd.DataFrame([{key: proposal[key] for key in FEATURES_B}])

    predictions = model_a.predict(row_a)[0]
    violations = dict(zip(VIOLATION_LABELS, [bool(value) for value in predictions]))
    utility_probability = float(model_b.predict_proba(row_b)[0][1])

    required = required_setbacks(proposal["height_proposed_m"])
    checks = {
        "far_violation": proposal["far_proposed"] > proposal["far_permissible"],
        "height_violation": proposal["height_proposed_m"] > proposal["height_permissible_m"],
        "setback_violation": (
            proposal["setback_front_m"] < required["front"]
            or proposal["setback_rear_m"] < required["rear"]
            or proposal["setback_side_m"] < required["side"]
        ),
        "coverage_violation": proposal["ground_coverage_pct"] > GROUND_COVERAGE_CAP[proposal["zone_type"]],
        "parking_violation": proposal["parking_provided_ecs"] < proposal["parking_required_ecs"],
    }

    reasons = []
    if violations["far_violation"]:
        reasons.append(
            f"Proposed FAR {proposal['far_proposed']} is above the estimated screening benchmark {proposal['far_permissible']}."
        )
    if violations["height_violation"]:
        reasons.append(
            f"Proposed height {proposal['height_proposed_m']} m is above the estimated screening benchmark {proposal['height_permissible_m']} m."
        )
    if violations["setback_violation"]:
        reasons.append(
            "Setbacks are below the estimated requirements "
            f"(front {required['front']} m, rear {required['rear']} m, side {required['side']} m)."
        )
    if violations["coverage_violation"]:
        reasons.append(
            f"Ground coverage is above the model's estimated {GROUND_COVERAGE_CAP[proposal['zone_type']]}% screening cap for this zone."
        )
    if violations["parking_violation"]:
        reasons.append(
            f"Parking provided ({proposal['parking_provided_ecs']} ECS) is below "
            f"the model's estimated benchmark ({proposal['parking_required_ecs']} ECS)."
        )
    if utility_probability >= 0.5:
        reasons.append(
            f"Predicted underground utility-conflict risk is {utility_probability:.0%}."
        )
    if not reasons:
        reasons.append("No planning flags or elevated utility risk were returned by the screening model.")

    predicted_violation = any(violations.values())
    feasible = not predicted_violation and utility_probability < 0.5
    bns_analysis = evaluate_bns_compliance(violations, utility_probability, proposal)

    return {
        "proposal": proposal,
        "violations": violations,
        "rule_checks": checks,
        "utility_conflict_probability": utility_probability,
        "feasible": feasible,
        "decision": "No screening flags returned" if feasible else "Review recommended",
        "reasons": reasons,
        "bns_legal_analysis": bns_analysis,
    }


BNS_REVIEW_DATABASE = {
    "far_violation": {
        "title": "Floor area ratio (FAR) needs planning review",
        "description": "The screening model flagged the proposed FAR against its estimated benchmark. Confirm the applicable local development control rules and sanctioned plan.",
        "bns_sections": [
            {
                "section": "BNS §223",
                "title": "Disobedience to an order duly promulgated by a public servant",
                "legal_rationale": "Relevant only if a competent public servant issued a lawful, applicable order and the person knowingly disobeyed it. A FAR screening flag alone does not establish this offence.",
            },
            {
                "section": "BNS §§270, 292–293",
                "title": "Public nuisance; punishment or continuation after an injunction",
                "legal_rationale": "Requires facts meeting the statutory ingredients of a public nuisance; sections 292 or 293 depend on the specific conduct and, for section 293, an injunction. Excess FAR by itself is not enough.",
            },
        ],
        "special_acts": ["Applicable state/UT development control regulations and master plan", "Applicable municipal building and planning law", "NBC 2016, where adopted or applicable"],
        "actions_taken": ["Have a registered architect or planner recalculate permitted FAR for the exact plot and zone.", "Compare the proposal with the sanctioned plan and current local development rules.", "Request written guidance from the local planning authority before construction or revision."],
    },
    "height_violation": {
        "title": "Proposed building height needs safety and planning review",
        "description": "The screening model flagged the proposed height against its estimated benchmark. Confirm local height controls, fire access, and any site-specific clearance requirements.",
        "bns_sections": [
            {
                "section": "BNS §125",
                "title": "Act endangering life or personal safety of others",
                "legal_rationale": "Requires evidence of a rash or negligent act that endangers life or personal safety. A modelled height exceedance alone does not establish endangerment.",
            },
            {
                "section": "BNS §290",
                "title": "Negligent conduct in pulling down, repairing, or constructing buildings",
                "legal_rationale": "Applies to specified negligent or knowing omissions in construction that create probable danger to human life. It is not a general penalty for exceeding a height limit.",
            },
        ],
        "special_acts": ["Applicable state/UT building and fire-safety rules", "NBC 2016, where adopted or applicable", "Airports Authority of India height clearance, if the site falls within a regulated area"],
        "actions_taken": ["Ask a licensed architect and structural/fire-safety professional to verify the height and access design.", "Check applicable fire NOC, aviation clearance, and municipal approval requirements.", "Revise the design or obtain the required written approval before proceeding."],
    },
    "setback_violation": {
        "title": "Setbacks need boundary and access review",
        "description": "The screening model flagged one or more proposed setbacks against an estimated requirement. A surveyed boundary and local rules are needed to confirm the result.",
        "bns_sections": [
            {
                "section": "BNS §285",
                "title": "Danger or obstruction in a public way or line of navigation",
                "legal_rationale": "Potentially relevant only where an act or omission actually causes danger, obstruction, or injury in a public way. A setback shortfall within a private plot is not by itself this offence.",
            },
            {
                "section": "BNS §125",
                "title": "Act endangering life or personal safety of others",
                "legal_rationale": "Requires facts showing a rash or negligent act that endangers life or personal safety; an estimated setback shortfall alone is insufficient.",
            },
        ],
        "special_acts": ["Applicable state/UT development control regulations", "Municipal boundary, road-widening, and fire-access requirements", "NBC 2016, where adopted or applicable"],
        "actions_taken": ["Commission a licensed boundary/topographic survey and confirm the road right-of-way.", "Recheck all side, rear, and front margins against the local rules for this plot.", "Keep required fire access and public paths clear; submit revised drawings for municipal review."],
    },
    "coverage_violation": {
        "title": "Ground coverage needs planning and drainage review",
        "description": "The screening model flagged the proposed footprint against an estimated zone cap. Verify the applicable local rule and any plot-specific exclusions.",
        "bns_sections": [
            {
                "section": "BNS §§270, 292–293",
                "title": "Public nuisance; punishment or continuation after an injunction",
                "legal_rationale": "These provisions require additional facts establishing a public nuisance and the statutory conditions. A ground-coverage percentage above a screening benchmark alone does not establish a BNS offence.",
            },
        ],
        "special_acts": ["Applicable state/UT development control regulations and municipal building law", "Local stormwater, recharge, and open-space rules", "NBC 2016, where adopted or applicable"],
        "actions_taken": ["Verify the permitted coverage with the local development control rules.", "Have an architect check the footprint, mandatory open space, and stormwater provisions.", "Revise the footprint or seek a written planning determination before construction."],
    },
    "parking_violation": {
        "title": "Parking provision needs local rule review",
        "description": "The model flagged a possible shortfall against an estimated parking benchmark. Required spaces vary by local rules and the approved use.",
        "bns_sections": [
            {
                "section": "BNS §285",
                "title": "Danger or obstruction in a public way or line of navigation",
                "legal_rationale": "Potentially relevant only if an act or omission causes actual danger, obstruction, or injury in a public way. A parking shortfall alone does not establish this offence.",
            },
        ],
        "special_acts": ["Applicable municipal parking and development control regulations", "Approved occupancy/use conditions and project-specific approval"],
        "actions_taken": ["Confirm the required parking ratio for the approved use and current local rules.", "Update the parking schedule and circulation plan with a licensed architect.", "Do not assume public-street parking can make up a required on-site provision."],
    },
    "utility_conflict": {
        "title": "Underground utility risk needs field verification",
        "description": "The utility model returned an elevated conflict-risk score. This is a probability estimate and does not confirm that a utility is present or damaged.",
        "bns_sections": [
            {
                "section": "BNS §324",
                "title": "Mischief involving damage to property",
                "legal_rationale": "Requires property damage and the statutory intent or knowledge described in the section. A predicted conflict, without damage and supporting facts, is not an offence.",
            },
            {
                "section": "BNS §125",
                "title": "Act endangering life or personal safety of others",
                "legal_rationale": "Potentially relevant only where evidence establishes a rash or negligent act that endangers life or personal safety; a model risk score alone does not do so.",
            },
        ],
        "special_acts": ["Utility-owner excavation permit and clearance requirements", "Applicable electricity, gas, water, and municipal utility rules"],
        "actions_taken": ["Obtain current utility-owner records and written clearances before excavation.", "Arrange a competent subsurface utility survey (such as GPR) and mark services on the ground.", "Use trial pits and an approved excavation method where required; stop work and notify the utility owner if a service is exposed or damaged."],
    },
}


def evaluate_bns_compliance(violations, utility_probability, proposal):
    """Return planning flags and conditional BNS references, not criminal findings."""
    review_items = []
    for key in ["far_violation", "height_violation", "setback_violation", "coverage_violation", "parking_violation"]:
        if violations.get(key):
            review_items.append({"violation_key": key, **BNS_REVIEW_DATABASE[key]})

    if utility_probability >= 0.5:
        review_items.append({"violation_key": "utility_conflict", **BNS_REVIEW_DATABASE["utility_conflict"]})

    flagged_count = len(review_items)
    return {
        "has_legal_liability": False,
        "has_review_items": bool(review_items),
        "status": f"{flagged_count} planning item(s) need verification" if review_items else "No screening flags returned",
        "total_infringements": flagged_count,
        "summary": (
            f"The screening returned {flagged_count} planning or safety item(s) to verify. None is a confirmed offence; BNS references below apply only if their statutory elements are supported by evidence."
            if review_items
            else "The screening returned no planning flags. It cannot determine whether any BNS offence has occurred."
        ),
        "infringements": review_items,
        "statutory_statutes": [
            "Bharatiya Nyaya Sanhita, 2023 (conditional references only)",
            "Applicable state/UT and municipal planning law",
            "National Building Code of India 2016, where adopted or applicable",
        ],
        "disclaimer": "This automated screening is not legal advice, a charge, or a finding that an offence occurred. Verify the current law and facts with the competent authority or a qualified advocate.",
    }
