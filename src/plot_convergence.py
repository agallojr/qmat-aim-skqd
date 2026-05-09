#!/usr/bin/env python3
"""Plot SQD energy convergence from a sweep run directory.

Usage:
    python src/plot_convergence.py <run_dir>

Example:
    python src/plot_convergence.py ~/output/2026-05-09/_63bffb51
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_cases(run_dir: Path) -> list[dict]:
    """Discover case subdirectories and load their data."""
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
        label = sub.name
        shots = None
        exact_energy = None
        params = {}
        if params_files:
            with open(params_files[0], "r", encoding="utf-8") as f:
                params = json.load(f)
            shots = params.get("--shots")
            kd = params.get("--krylov-dim")
            # Short label: k=<dim>, <shots> or statevector
            parts = []
            if kd is not None:
                parts.append(f"k={kd}")
            if shots is not None and shots > 0:
                if shots >= 1000:
                    parts.append(f"{shots // 1000}k shots")
                else:
                    parts.append(f"{shots} shots")
            elif shots == 0:
                parts.append("sv")
            label = ", ".join(parts) if parts else sub.name

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

    M = params.get("--num-imp-orbs")
    B = params.get("--num-bath-per-imp")
    if M is not None and B is not None:
        n_orbs = M * (1 + B)
        n_qubits = 2 * n_orbs
        n_elec = n_orbs  # half-filled
        dim = comb(n_orbs, n_elec // 2) ** 2
        lines.append(f"{M} imp × {B} bath = {n_orbs} orbitals, {n_qubits} qubits")
        lines.append(f"Hilbert dim (Nα,Nβ) = C({n_orbs},{n_elec//2})² = {dim:,}")

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


def plot_convergence(cases: list[dict], run_dir: Path) -> None:
    """Create convergence plot with physics subtitle and results table below."""

    # Collect exact energy and reference params (same for all cases)
    exact = None
    ref_params = {}
    for c in cases:
        if c["exact_energy"] is not None:
            exact = c["exact_energy"]
        if c["params"]:
            ref_params = c["params"]

    # --- Build title + physics header ---
    M = ref_params.get("--num-imp-orbs")
    B = ref_params.get("--num-bath-per-imp")
    sys_label = ref_params.get("--system-label")
    if M is not None and B is not None:
        n_orbs = M * (1 + B)
        title = f"SQD Convergence — {M}-orbital AIM  ({2*n_orbs}q)"
        if sys_label:
            title += f"  [{sys_label}]"
    else:
        title = "SQD Convergence"
        if sys_label:
            title += f"  [{sys_label}]"

    phys = _physics_text(ref_params)
    # Combine title and physics into one block
    header = title + "\n" + phys if phys else title

    n_header_lines = header.count("\n") + 1
    top_margin = 0.92 - 0.022 * n_header_lines

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.30},
    )
    fig.subplots_adjust(top=top_margin)

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
        ax1.plot(
            iters, energies, "o-",
            color=colors[i], label=c["label"], markersize=5, lw=1.5,
        )

    if exact is not None:
        ax1.axhline(
            exact, color="k", ls="--", lw=1.0,
            label=f"FCI exact = {exact:.6f} eV",
        )

    ax1.set_ylabel("Ground-state energy (eV)")
    ax1.set_xlabel("SQD iteration")
    ax1.ticklabel_format(useOffset=False, style="plain")
    ax1.legend(fontsize=8, loc="upper right")

    max_iter = max(len(c["energies"]) for c in cases)
    ax1.set_xticks(range(1, max_iter + 1))

    # --- Panel 2: Results table ---
    ax2.axis("off")
    table_data = []
    col_labels = ["Case", "Final E (eV)", "ΔE (meV)", "ΔE (%)", "Iters"]
    for c in cases:
        final_e = c["energies"][-1]
        if exact:
            err_eV = abs(final_e - exact)
            err_meV = err_eV * 1000.0
            err_pct = abs(err_eV / exact) * 100
            mev_str = f"{err_meV:.1f}"
            pct_str = f"{err_pct:.4f}"
        else:
            mev_str = "—"
            pct_str = "—"
        table_data.append([
            c["label"],
            f"{final_e:.6f}",
            mev_str,
            pct_str,
            str(len(c["energies"])),
        ])
    if exact is not None:
        table_data.append([
            "FCI (exact)", f"{exact:.6f}", "—", "—", "—",
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
    # Highlight rows with < 20% error
    if exact is not None:
        for i, c in enumerate(cases):
            final_e = c["energies"][-1]
            err_pct = abs((final_e - exact) / exact) * 100
            if err_pct < 20.0:
                for j in range(len(col_labels)):
                    table[i + 1, j].set_facecolor("#D5F5D5")
        # FCI row
        fci_row = len(table_data)
        for j in range(len(col_labels)):
            table[fci_row, j].set_facecolor("#E2EFDA")

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
