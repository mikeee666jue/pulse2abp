"""Print the result tables of the README from the metric files in results/.

    python scripts/results_table.py

Standard library only, no data and no GPU. Every number in the README tables is a line of this output.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
QUANTITIES = ("sbp", "dbp", "map")


def load(name):
    return json.loads((R / f"{name}.json").read_text())


def num(v, digits=2):
    if v is None or v != v:
        return "–"
    s = f"{v:.{digits}f}"
    return s.lstrip("-") if float(s) == 0 else s


def table(title, header, rows, text_columns=1):
    """Markdown table; the first `text_columns` columns are left-aligned, the numbers right-aligned."""
    print(f"\n{title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" if i < text_columns else "---:" for i in range(len(header))) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


def cells(m, draws=True, width=False):
    """MAE of SBP, DBP and MAP; then CRPS and 90% coverage of MAP for arms that return draws."""
    out = [num(m[q]["mae"]) for q in QUANTITIES]
    out += [num(m["map"]["crps"]), num(m["map"]["cov90"], 3)] if draws else ["–", "–"]
    if width:
        out.append(num(m["map"]["width90"], 1) if draws else "–")
    return out


def first_round():
    b0 = load("b0")
    rows = [["B0 climatology", "nothing"] + cells(b0["climatology"]) + [num(b0["climatology"]["hypo"]["auroc"], 3)],
            ["B0 persistence", "calibration"] + cells(b0["persistence"]) + [num(b0["persistence"]["hypo"]["auroc"], 3)]]
    for name, label, given, draws in (("b1", "B1 regression", "pulse", False),
                                      ("m1", "M1 diffusion", "pulse", True),
                                      ("m3", "M3 diffusion", "pulse + calibration", True)):
        m = load(name)["metrics"]
        rows.append([label, given] + cells(m, draws) + [num(m["hypo"]["auroc"], 3)])
    n, k = b0["n"], b0["config"]["k"]
    table(f"Table 1. First round: {n:,} test segments, {k} draws per segment, 50 sampling steps (mmHg)",
          ["Arm", "Given", "SBP MAE", "DBP MAE", "MAP MAE", "MAP CRPS", "MAP 90% coverage", "Hypotension AUROC"], rows,
          text_columns=2)

    r = load("reanalysis_map")
    rows = []
    for name, label in (("b1", "B1 regression"), ("m1", "M1 diffusion"), ("m3", "M3 diffusion")):
        d, w = r["distribution"][name], r["within_between"][name]
        rows.append([label, num(d["mae"]), num(d["crps_constant_bar"]), num(d.get("crps_own_draws")),
                     num(w["pulse_following"]), num(w["calibration_following"]), num(w["between_r2"])])
    table(f"Table 2. Same {r['segments']:,} segments, MAP, read again from the saved draws ({r['patients']} patients)",
          ["Arm", "MAE", "CRPS, point + fixed bar", "CRPS, own draws", "Pulse-following", "Calibration-following",
           "Between-patient R2"], rows)
    print("\n95% intervals from a bootstrap over patients:\n")
    for key, v in r["bootstrap"].items():
        sign = "" if key.startswith("coverage") else "+"      # differences carry a sign, coverages do not
        print(f"    {key:42s} {v['value']:{sign}.2f}  [{v['lo']:{sign}.2f}, {v['hi']:{sign}.2f}]")


def second_round():
    arms = (("b1_s15k", "B1 regression", 0, False),
            ("m1_s15k", "M1 diffusion", 0, True), ("m1_s15k_seed1", "M1 diffusion", 1, True),
            ("m1_linear_s15k", "M1 flow matching", 0, True), ("m1_linear_s15k_seed1", "M1 flow matching", 1, True),
            ("m2_frozen_s15k", "M2 PaPaGei, frozen", 0, True),
            ("m2_lora_s15k", "M2 PaPaGei, LoRA", 0, True), ("m2_lora_s15k_seed1", "M2 PaPaGei, LoRA", 1, True),
            ("m2_full_s15k", "M2 PaPaGei, fully fine-tuned", 0, True),
            ("m2_scratch_s15k", "M2 same encoder, random start", 0, True),
            ("m3_s15k", "M3 diffusion + calibration", 0, True))
    rows, n, k = [], None, None
    for name, label, seed, draws in arms:
        d = load(f"n7200_k19_s50/{name}")
        assert d["train_args"]["seed"] == seed and d["train_args"]["steps"] == 15000, name
        n, k = d["n"], d["config"]["k"]
        rows.append([label, str(seed)] + cells(d["metrics"], draws))
    table(f"Table 3. Second round, 15,000 training steps: {n:,} test segments, {k} draws, 50 sampling steps (mmHg)",
          ["Arm", "Seed", "SBP MAE", "DBP MAE", "MAP MAE", "MAP CRPS", "MAP 90% coverage"], rows)


def input_tests():
    tests = (("m3", "as trained"),
             ("m3_far", "stale calibration (`--cal far`)"),
             ("m3_other", "another patient's calibration (`--cal other`)"),
             ("m3_noise", "noisy pulse (`--ppg noise:0.5`)"),
             ("n7200_k19_s50/m3_nocal", "no calibration (`--cal none`)"),
             ("n7200_k19_s50/m3_w0", "pulse switched off (`--w-ppg 0`)"),
             ("n7200_k19_s50/m3_w2", "pulse weight doubled (`--w-ppg 2`)"))
    rows = []
    for name, label in tests:
        d = load(name)
        rows.append([label, f"{d['n']:,}"] + cells(d["metrics"], width=True)[2:])
    table("Table 4. M3 (first-round checkpoint) when its inputs are removed, corrupted or wrong, MAP (mmHg)",
          ["Test", "Segments", "MAE", "CRPS", "90% coverage", "90% interval width"], rows)

    rows = []
    for steps, name in ((25, "n7200_k19_s25/m3"), (100, "n7200_k19_s100/m3")):
        d = load(name)
        assert d["config"]["steps"] == steps
        rows.append([str(steps), f"{d['n']:,}"] + cells(d["metrics"], width=True)[2:])
    table("Table 5. M3 with 25 and 100 sampling steps instead of 50, MAP (mmHg)",
          ["Sampling steps", "Segments", "MAE", "CRPS", "90% coverage", "90% interval width"], rows)


if __name__ == "__main__":
    first_round()
    second_round()
    input_tests()
