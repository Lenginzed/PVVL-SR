from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript" / "taes_pvvl_sr_v01"


def read_all_tex() -> str:
    parts = [MS / "main.tex"]
    parts.extend(sorted((MS / "sections").glob("*.tex")))
    parts.extend(sorted((MS / "tables").glob("*.tex")))
    parts.extend([MS / "acronyms.tex", MS / "notation.tex"])
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in parts if p.exists())


def main() -> None:
    checks = []
    text = read_all_tex()
    main = (MS / "main.tex").read_text(encoding="utf-8", errors="ignore")

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    add("main.tex exists", (MS / "main.tex").exists())
    for i in range(1, 9):
        section = MS / "sections" / f"{i:02d}_" 
    for rel in [
        "sections/01_introduction.tex", "sections/02_related_work.tex",
        "sections/03_problem_formulation.tex", "sections/04_method.tex",
        "sections/05_experiments.tex", "sections/06_results.tex",
        "sections/07_discussion.tex", "sections/08_conclusion.tex",
    ]:
        add(f"{rel} exists and input", (MS / rel).exists() and rel.replace(".tex", "") in main)

    for fig in [
        "fig1_method_architecture.pdf", "fig2_label_quality_improvement.pdf",
        "fig3_surrogate_model_accuracy.pdf", "fig4_mixed_evaluation_performance.pdf",
        "fig5_300k_to_500k_trend.pdf", "fig6_sample_efficiency.pdf",
        "fig7_representative_trajectories.pdf", "fig8_dodgemissile_preliminary.pdf",
    ]:
        add(f"figure {fig} exists", (MS / "figures" / fig).exists())

    for table in [
        "table_i_mixed.tex", "table_ii_offensive.tex", "table_iii_unseen.tex",
        "table_iv_ablation.tex", "table_v_dodge.tex",
    ]:
        add(f"table {table} exists", (MS / "tables" / table).exists())

    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", text))
    missing_refs = sorted(refs - labels)
    add("all refs have labels", not missing_refs, ", ".join(missing_refs))

    danger = ["fully outperforms", "solves air combat", "solves UAV air combat", "state-of-the-art", "proves missile combat capability", "VLM controls"]
    danger_hits = [d for d in danger if d.lower() in text.lower()]
    add("dangerous overclaim scan", not danger_hits, ", ".join(danger_hits))

    add("PPO-Phys threat-control advantage mentioned", "PPO-Phys has lower enemy-threat exposure" in text or "physical shaping remains a strong threat-control baseline" in text)
    add("DodgeMissile preliminary mentioned", "preliminary" in text and "DodgeMissile" in text)
    add(
        "NoWeapon limitation mentioned",
        "NoWeapon" in text
        and ("does not validate full missile-combat" in text or "do not validate full missile-combat" in text),
    )
    add("optimizer reset caveat mentioned", "optimizer state is reset" in text or "resets optimizer state" in text)
    add("Zhendong Li not marked IEEE member", "Zhendong Li}\n\\member" not in main and "Student Member" not in main)
    add("Hui Li Senior Member", "Hui Li}\n\\member{Senior Member, IEEE}" in main)
    add("Hui Li corresponding author", "Corresponding author: Hui Li" in main)

    cite_placeholders = sorted(c for c in set(re.findall(r"\\needcite\{([^}]+)\}", text)) if c != "...")
    add("citation placeholders intentionally present", bool(cite_placeholders), f"{len(cite_placeholders)} placeholders")

    lines = ["# Manuscript Checklist", ""]
    required_checks = [(name, ok, detail) for name, ok, detail in checks if name != "citation placeholders intentionally present"]
    status = "PASS" if all(ok for _, ok, _ in required_checks) else "CHECK"
    lines.append(f"Overall status: {status}")
    lines.append("")
    lines.append("| Check | Status | Detail |")
    lines.append("|---|---|---|")
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.append("")
    lines.append("## Citation TODO Placeholders")
    for c in cite_placeholders:
        lines.append(f"- {c}")
    (MS / "manuscript_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
