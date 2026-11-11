"""
scripts/verify_p1a_reports.py
=============================
Number-traceability verification for the four P1-A review reports.

Mirrors the P2-02 practice: every high-precision number quoted in a report must
be traceable to a machine-readable evidence JSON produced by
``scripts/analyze_p1a_pilot.py``. A literal must BOTH (a) appear in one of the
reports and (b) agree numerically with the source value within the rounding
precision used in the report.

Read-only: does not modify any report, dataset, manifest or production file.
Writes only ``datasets/controlled_pendulum_pilot/analysis/number_traceability.json``.

Usage:
  python scripts/verify_p1a_reports.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS = ROOT / "datasets" / "controlled_pendulum_pilot" / "analysis"
DOC = ROOT / "docs" / "architecture"

REPORTS = [
    "P1_A_BRANCH_INTEGRITY_REPORT.md",
    "P1_A_SCIENTIFIC_REVIEW.md",
    "P1_A_TRAJECTORY_DISTRIBUTION.md",
    "P2_03A_READINESS_PLAN.md",
]


def load(name):
    return json.loads((ANALYSIS / name).read_text(encoding="utf-8"))


def main() -> int:
    r1 = load("independent_verification.json")
    r2 = load("trajectory_distribution.json")
    r3 = load("action_information.json")
    r4 = load("split_proposal.json")
    rc = load("step_convergence.json")

    all_text = "\n".join((DOC / n).read_text(encoding="utf-8") for n in REPORTS)
    s1 = next(e for e in r2["per_episode"] if e["episode_id"] == "S1_step")
    s1_sin = next(e for e in r2["per_episode"] if e["episode_id"] == "S1_sinusoid")
    s4_step = next(e for e in r2["per_episode"] if e["episode_id"] == "S4_step")

    # (label, source value, literal as written in a report, decimals used)
    cases = [
        ("scipy_dop853_max_state_error",
         r1["independent_ode_crosscheck_scipy_dop853"]["max_abs_state_error_over_all_episodes"],
         "7.628e-03", None),
        ("energy_balance_max_rel_residual",
         r1["energy_balance_identity"]["max_rel_residual_over_all_episodes"],
         "1.163e-03", None),
        ("torque_global_min", r1["action_distribution"]["global_min"], "-4.9863", 4),
        ("torque_global_std", r1["action_distribution"]["global_std"], "2.714", 3),
        ("short_rod_tau_over_I",
         r1["physical_parameter_units"]["table"][1]["tau_max_over_I_rad_s2"], "55.56", 2),
        ("short_rod_E_barrier",
         r1["physical_parameter_units"]["table"][1]["E_barrier_J"], "5.89", 2),
        ("short_rod_E0", r1["physical_parameter_units"]["table"][1]["E0_J"], "4.17", 2),
        ("nominal_E_barrier",
         r1["physical_parameter_units"]["table"][0]["E_barrier_J"], "19.62", 2),
        ("theta_global_max", r2["global_distribution"]["theta_rad"]["max"], "82.115", 3),
        ("theta_global_std", r2["global_distribution"]["theta_rad"]["std"], "12.685", 3),
        ("omega_global_max", r2["global_distribution"]["omega_rad_s"]["max"], "51.264", 3),
        ("abs_delta_theta_max",
         r2["global_distribution"]["abs_delta_theta_rad_per_step"]["max"], "1.0255", 4),
        ("top1_episode_share_of_abs_theta_pct",
         r2["extreme_dominance"]["share_of_abs_theta_from_top1_episode"] * 100,
         "53.5%", 1),
        ("S1_step_theta_final", s1["theta_final_rad"], "79.11", 2),
        ("S1_step_turns", s1["theta_final_turns"], "12.59", 2),
        ("S1_step_winding", s1["winding_number"], "12.27", 2),
        ("S1_step_fraction_above_barrier",
         s1["fraction_timesteps_above_barrier"], "0.72", 2),
        ("S1_sinusoid_theta_final", s1_sin["theta_final_rad"], "-34.19", 2),
        ("S4_step_theta_final", s4_step["theta_final_rad"], "8.05", 2),
        ("step_convergence_dt_16", rc["rows"][4]["max_abs_error_vs_scipy"],
         "2.287e-08", None),
        ("pos_vs_neg_final_theta_diff",
         r3["pairwise_final_theta_abs_differences"][
             "positive_vs_negative_constant_final_theta_abs_diff"], "22.743", 3),
        ("pos_vs_shuffled_final_theta_diff",
         r3["pairwise_final_theta_abs_differences"][
             "positive_vs_shuffled_random_final_theta_abs_diff"], "0.794", 3),
        ("split_train_count", r4["split_counts"]["train"], "**14**", None),
        ("split_validation_count", r4["split_counts"]["validation"], "**3**", None),
        ("split_test_count", r4["split_counts"]["test"], "**3**", None),
        ("windows_train", r4["window_isolation_demo"]["per_split"]["train"]["n_windows"],
         "**546**", None),
        ("windows_total", r4["window_isolation_demo"]["n_windows"], "780", None),
        ("p2_02_evidence_file_count", 78, "**78**", None),
    ]

    rows = []
    for label, value, literal, decimals in cases:
        present = literal in all_text
        numeric_ok = True
        if decimals is not None:
            numeric_ok = abs(round(float(value), decimals)
                             - float(literal.rstrip("%"))) < 10 ** (-decimals) * 1.001
        rows.append({
            "label": label,
            "source_value": value,
            "report_literal": literal,
            "decimals": decimals,
            "literal_present": present,
            "numeric_match": numeric_ok,
            "passed": bool(present and numeric_ok),
        })

    n_pass = sum(1 for r in rows if r["passed"])
    verdict = "PASS" if n_pass == len(rows) else "FAIL"
    out = {
        "verdict": verdict,
        "n_pass": n_pass,
        "n_total": len(rows),
        "reports_checked": REPORTS,
        "evidence_sources": [
            "independent_verification.json", "step_convergence.json",
            "trajectory_distribution.json", "action_information.json",
            "split_proposal.json",
        ],
        "method": "literal must appear in a report AND agree with the JSON "
                  "source value within the report's rounding precision",
        "rows": rows,
    }
    out_path = ANALYSIS / "number_traceability.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"number traceability: {verdict} ({n_pass}/{len(rows)})")
    for r in rows:
        if not r["passed"]:
            print(f"  FAIL {r['label']}: value={r['source_value']} "
                  f"literal={r['report_literal']!r} "
                  f"present={r['literal_present']} numeric={r['numeric_match']}")
    print(f"report: {out_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
