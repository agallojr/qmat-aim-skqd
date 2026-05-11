#!/usr/bin/env python3
"""Plot SQD energy convergence from a sweep run directory.

Usage:
    python src/plot_convergence.py <run_dir>

Example:
    python src/plot_convergence.py ~/output/2026-05-09/_63bffb51
"""

import datetime as _dt
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WF_DIR_RE = re.compile(r"^_[0-9a-f]+$")


def provenance(run_dir: Path) -> tuple[str, str]:
    """Extract (wf_id, date) from a run directory.

    Convention: ~/output/<YYYY-MM-DD>/_<hex-wf-id>/. Falls back to the run
    directory's literal name and mtime when the path doesn't match.
    """
    name = run_dir.name
    parent = run_dir.parent.name

    wf_id = name if _WF_DIR_RE.match(name) else name
    if _DATE_DIR_RE.match(parent):
        date_str = parent
    else:
        date_str = _dt.date.fromtimestamp(run_dir.stat().st_mtime).isoformat()
    return wf_id, date_str


def _load_group_map(run_dir: Path) -> dict[str, str]:
    """Read _final_postproc.json and return {case_dir_name: group_name}.

    The sweep writes parallel arrays case_dirs[i] ↔ groups[i]; both may
    be absent on standalone runs, in which case an empty map is returned.
    """
    fp = run_dir / "_final_postproc.json"
    if not fp.exists():
        return {}
    with open(fp, "r", encoding="utf-8") as f:
        meta = json.load(f)
    case_dirs = meta.get("case_dirs") or []
    groups = meta.get("groups") or []
    return {Path(d).name: g for d, g in zip(case_dirs, groups)}


def load_cases(run_dir: Path) -> list[dict]:
    """Discover case subdirectories and load their data."""
    group_map = _load_group_map(run_dir)
    cases = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        energy_path = sub / "energy_history.json"
        if not energy_path.exists():
            continue

        with open(energy_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        # Get case label and params from sweep metadata
        params_files = list(sub.glob("q8020_params_*.json"))
        case_files = list(sub.glob("q8020_case_*.json"))

        # Try to get group from multiple sources
        group = group_map.get(sub.name)

        # Fallback 1: look in case metadata
        if not group and case_files:
            with open(case_files[0], "r", encoding="utf-8") as f:
                case_meta = json.load(f)
            group = case_meta.get("group")

        shots = None
        exact_energy = None
        params = {}
        if params_files:
            with open(params_files[0], "r", encoding="utf-8") as f:
                params = json.load(f)
            shots = params.get("--shots")

        # Fallback 2: construct group from krylov-dim if still missing
        if not group and params:
            kd = params.get("--krylov-dim")
            if kd:
                group = f"k{kd}"

        # Fallback 3: try to extract from directory name
        if not group:
            # Try to extract group from directory name (e.g., "k4_something" -> "k4")
            dirname = sub.name
            match = re.match(r"^([a-zA-Z0-9_]+?)(?:_\d+|$)", dirname)
            if match:
                group = match.group(1)

        # Construct label as groupName_shots (group should always exist from TOML)
        if shots is not None and shots > 0:
            if shots >= 1000:
                shot_str = f"{shots // 1000}k"
            else:
                shot_str = f"{shots}sh"
        elif shots == 0:
            shot_str = "sv"
        else:
            shot_str = None

        label = f"{group}_{shot_str}" if shot_str else str(group)

        # Get exact energy and timing from results fragment
        results_files = list(sub.glob("q8020_results_*.json"))
        results = {}
        if results_files:
            with open(results_files[0], "r", encoding="utf-8") as f:
                results = json.load(f)
            exact_energy = results.get("exact_energy")

        # Get circuit stats
        exec_files = list(sub.glob("q8020_exec_stats_*.json"))
        exec_stats = {}
        if exec_files:
            with open(exec_files[0], "r", encoding="utf-8") as f:
                exec_stats = json.load(f)

        # Fallback: read run_args from code fragment (standalone runs)
        if not params:
            code_files = list(sub.glob("q8020_code_*.json"))
            if code_files:
                with open(code_files[0], "r", encoding="utf-8") as f:
                    code_meta = json.load(f)
                run_args = code_meta.get("run_args", {})
                # Normalize to --flag style keys used by _physics_text
                params = {
                    f"--{k.replace('_', '-')}": v
                    for k, v in run_args.items()
                    if v is not None
                }

        cases.append({
            "label": label,
            "energies": history["energies"],
            "exact_energy": exact_energy,
            "shots": shots,
            "params": params,
            "exec_stats": exec_stats,
            "dir": sub,
        })

    return cases


def _physics_text(params: dict) -> str:
    """Build a multi-line annotation string from AIM parameters."""
    from math import comb
    lines = []

    sys_label = params.get("--system-label")
    if sys_label:
        lines.append(f"System: {sys_label}")

    M = params.get("--num-imp-orbs")
    B = params.get("--num-bath-per-imp")
    if M is not None and B is not None:
        n_orbs = M * (1 + B)
        n_qubits = 2 * n_orbs
        na = params.get("--n-electrons-alpha")
        nb = params.get("--n-electrons-beta")
        if na is None or nb is None:
            tot = params.get("--n-electrons")
            if tot is not None:
                na = (tot + 1) // 2
                nb = tot - na
            else:
                na = nb = n_orbs // 2
        dim = comb(n_orbs, na) * comb(n_orbs, nb)
        lines.append(f"{M} imp × {B} bath = {n_orbs} orbitals, {n_qubits} qubits")
        lines.append(
            f"Hilbert dim (Nα={na}, Nβ={nb}) = "
            f"C({n_orbs},{na})·C({n_orbs},{nb}) = {dim:,}"
        )

    U = params.get("--onsite")
    if U is not None:
        parts = [f"U = {U}"]
        Up = params.get("--U-prime")
        if Up is not None:
            parts.append(f"U′ = {Up}")
        Jh = params.get("--J-H")
        if Jh is not None:
            parts.append(f"J_H = {Jh}")
        lines.append("  ".join(parts) + "  eV")

    mu = params.get("--mu")
    V = params.get("--hybridization")
    if mu is not None and V is not None:
        lines.append(f"μ = {mu}  V = {V}  eV")

    cf = params.get("--crystal-field")
    if cf is not None:
        lines.append(f"ε_CF = {cf}")

    return "\n".join(lines)


_GLOSSARY = (
    "M = imp orbitals    B = bath sites per imp    "
    "k = Krylov dim (# time-evolved basis states)\n"
    "U = intra-orbital Coulomb (Hubbard)    "
    "U′ = inter-orbital Coulomb    "
    "J_H = Hund's exchange\n"
    "μ = chemical potential    "
    "V = impurity–bath hybridization    "
    "ε_CF = crystal-field splitting"
)


def plot_convergence(cases: list[dict], run_dir: Path) -> None:
    """Create convergence plot with physics subtitle and results table below."""

    # Per-case FCI (each case may have its own Hamiltonian — bath/nelec/etc.).
    # ref_params is any one case's params, used only for header text that's
    # shared across cases (M, B, U, ...).
    ref_params = {}
    for c in cases:
        if c["params"]:
            ref_params = c["params"]
    cases_share_fci = (
        len({round(c["exact_energy"], 9)
             for c in cases if c["exact_energy"] is not None}) <= 1
    )
    shared_exact = (
        cases[0]["exact_energy"]
        if cases_share_fci and cases and cases[0]["exact_energy"] is not None
        else None
    )

    # --- Build title + physics header ---
    M = ref_params.get("--num-imp-orbs")
    B = ref_params.get("--num-bath-per-imp")
    sys_label = ref_params.get("--system-label")
    if M is not None and B is not None:
        n_orbs = M * (1 + B)
        title = f"SQD Convergence — {M}-orbital AIM  ({2*n_orbs}q)"
    else:
        title = "SQD Convergence"

    wf_id, date_str = provenance(run_dir)
    provenance_line = f"wf {wf_id}  ·  {date_str}"
    if sys_label:
        provenance_line += f"  ·  [{sys_label}]"

    phys = _physics_text(ref_params)
    # Combine title, physics, and provenance into one block
    if phys:
        header = title + "\n" + phys + "\n" + provenance_line
    else:
        header = title + "\n" + provenance_line

    n_header_lines = header.count("\n") + 1
    top_margin = 0.92 - 0.022 * n_header_lines

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7.7),
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.30},
    )
    fig.subplots_adjust(top=top_margin, bottom=0.15, right=0.82)

    # --- Header box: title + physics (single element, no gap) ---
    fig.text(
        0.5, top_margin + 0.005, header,
        ha="center", va="bottom",
        fontsize=8, family="monospace",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.85),
    )

    # --- Panel 1: Energy convergence ---
    cmap = plt.get_cmap("Dark2")
    colors = [cmap(i / max(len(cases) - 1, 1)) for i in range(len(cases))]

    for i, c in enumerate(cases):
        iters = np.arange(1, len(c["energies"]) + 1)
        energies = np.array(c["energies"])
        ex = c["exact_energy"]
        # When cases share a single FCI we draw it once below; otherwise
        # tag each curve's legend entry with its own FCI for clarity.
        legend_label = c["label"]
        if not cases_share_fci and ex is not None:
            legend_label = f"{c['label']}  (FCI={ex:.4f})"
        ax1.plot(
            iters, energies, "o-",
            color=colors[i], label=legend_label, markersize=5, lw=1.5,
        )
        # Per-case FCI line, colored to match its curve. Skipped when
        # all cases share one FCI (single dashed line drawn below).
        if not cases_share_fci and ex is not None:
            ax1.axhline(ex, color=colors[i], ls=":", lw=0.9, alpha=0.7)

    if cases_share_fci and shared_exact is not None:
        ax1.axhline(
            shared_exact, color="k", ls="--", lw=1.0,
            label=f"FCI exact = {shared_exact:.6f} eV",
        )

    ax1.set_ylabel("Ground-state energy (eV)")
    ax1.set_xlabel("SQD iteration")
    ax1.ticklabel_format(useOffset=False, style="plain")
    ax1.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))

    max_iter = max(len(c["energies"]) for c in cases)
    ax1.set_xticks(range(1, max_iter + 1))

    # --- Panel 2: Results table (per-case FCI) ---
    ax2.axis("off")
    table_data = []
    col_labels = [
        "Case", "Final E (eV)", "FCI (eV)", "ΔE (meV)", "ΔE (%)", "Iters",
    ]
    for c in cases:
        final_e = c["energies"][-1]
        ex = c["exact_energy"]
        if ex is not None and abs(ex) > 1e-15:
            err_eV = abs(final_e - ex)
            err_meV = err_eV * 1000.0
            err_pct = abs(err_eV / ex) * 100
            mev_str = f"{err_meV:.1f}"
            pct_str = f"{err_pct:.4f}"
            fci_str = f"{ex:.6f}"
        else:
            mev_str = "—"
            pct_str = "—"
            fci_str = "—" if ex is None else f"{ex:.6f}"
        table_data.append([
            c["label"],
            f"{final_e:.6f}",
            fci_str,
            mev_str,
            pct_str,
            str(len(c["energies"])),
        ])
    if cases_share_fci and shared_exact is not None:
        table_data.append([
            "FCI (exact)", f"{shared_exact:.6f}",
            f"{shared_exact:.6f}", "—", "—", "—",
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight: green if within chemical accuracy (< 43 meV),
    # yellow if < 2% error, otherwise no color
    for i, c in enumerate(cases):
        ex = c["exact_energy"]
        if ex is None or abs(ex) < 1e-15:
            continue
        final_e = c["energies"][-1]
        err_eV = abs(final_e - ex)
        err_pct = abs((final_e - ex) / ex) * 100
        if err_eV < 0.043:  # Chemical accuracy = 43 meV
            color = "#D5F5D5"  # Green
        elif err_pct < 2.0:
            color = "#FFFACD"  # Yellow
        else:
            color = None
        if color:
            for j in range(len(col_labels)):
                table[i + 1, j].set_facecolor(color)
    if cases_share_fci and shared_exact is not None:
        fci_row = len(table_data)
        for j in range(len(col_labels)):
            table[fci_row, j].set_facecolor("#E2EFDA")

    # --- Glossary box at bottom ---
    fig.text(
        0.5, 0.01, _GLOSSARY,
        ha="center", va="bottom",
        fontsize=7, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="#F0F0F0",
                  ec="#999999", alpha=0.9),
    )

    out_path = run_dir / "convergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    if sys.stdout.isatty():
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/plot_convergence.py <run_dir_or_json>")
        sys.exit(1)

    arg = Path(sys.argv[1]).expanduser()

    # If argument is a JSON file (_final_postproc.json), extract run_dir
    if arg.is_file() and arg.suffix == ".json":
        with open(arg, "r", encoding="utf-8") as f:
            meta = json.load(f)
        run_dir = Path(meta["run_dir"])
    elif arg.is_dir():
        run_dir = arg
    else:
        print(f"Error: {arg} is not a directory or JSON file")
        sys.exit(1)

    cases = load_cases(run_dir)
    if not cases:
        print(f"No cases with energy_history.json found in {run_dir}")
        sys.exit(1)

    print(f"Found {len(cases)} cases:")
    for c in cases:
        e = c["energies"]
        print(f"  {c['label']}: {len(e)} iterations, "
              f"final = {e[-1]:.6f}")

    plot_convergence(cases, run_dir)


if __name__ == "__main__":
    main()
