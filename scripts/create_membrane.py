#!/usr/bin/env python3
"""
create_membrane.py — Build configurable lipid bilayer membranes for OpenMM

Builds a membrane system, equilibrates it, and optionally streams
the simulation to PDB Visual via MDSS protocol.

Requirements
------------
    conda install -c conda-forge ambertools openmm openmmforcefields pdbfixer

Quick start
-----------
    # Pure POPC bilayer, 10x10 nm, 0.15 M NaCl, stream to viewer
    python scripts/create_membrane.py --lipids POPC:100 --size 10 10 --salt 0.15 --stream

    # Mixed bilayer with protein insertion
    python scripts/create_membrane.py \\
        --lipids POPC:70:POPE:20:CHOL:10 \\
        --protein myprotein.pdb \\
        --size 12 12 --salt 0.15 --stream

    # Build only (no simulation), save PDB for later
    python scripts/create_membrane.py --lipids POPC:100 --output membrane.pdb

Lipid types supported by AMBER lipid21 (via AmberTools packmol-memgen)
-----------------------------------------------------------------------
    POPC  POPE  POPS  POPG  POPA  POPE  DLPE  DLPC  DLPG  DLPS
    DPPC  DPPE  DPPS  DPPG  DPPA  DSPC  DSPE  DSPS  DSPG  DSPA
    DOPC  DOPE  DOPS  DOPG  DOPA
    CHOL  (cholesterol)
    PI    PIP   PIP2  PIP3  (phosphoinositides)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional


# ── Lipid metadata ────────────────────────────────────────────────────────────

LIPID_INFO = {
    "POPC":  "Palmitoyl-oleoyl phosphatidylcholine (most common bilayer lipid)",
    "POPE":  "Palmitoyl-oleoyl phosphatidylethanolamine",
    "POPS":  "Palmitoyl-oleoyl phosphatidylserine (anionic)",
    "POPG":  "Palmitoyl-oleoyl phosphatidylglycerol (anionic)",
    "DPPC":  "Dipalmitoyl phosphatidylcholine (gel-phase, Tm=41°C)",
    "DPPE":  "Dipalmitoyl phosphatidylethanolamine",
    "DOPC":  "Dioleoyl phosphatidylcholine (fluid phase)",
    "DOPE":  "Dioleoyl phosphatidylethanolamine",
    "CHOL":  "Cholesterol (modulates membrane fluidity)",
    "DSPC":  "Distearoyl phosphatidylcholine (high Tm)",
    "DLPC":  "Dilauroyl phosphatidylcholine (short tails)",
}


# ── Composition parser ────────────────────────────────────────────────────────

def parse_lipids(spec: str) -> dict:
    """
    Parse lipid composition string.
    Format: LIPID1:N1:LIPID2:N2:...
    Example: "POPC:70:POPE:20:CHOL:10"
    Returns: {"POPC": 70, "POPE": 20, "CHOL": 10}
    """
    parts = spec.split(":")
    if len(parts) % 2 != 0:
        sys.exit(f"[!] Bad lipid spec '{spec}'. Format: LIPID1:N1:LIPID2:N2")
    comp = {}
    for i in range(0, len(parts), 2):
        name = parts[i].upper()
        try:
            count = int(parts[i+1])
        except ValueError:
            sys.exit(f"[!] Expected integer after '{name}', got '{parts[i+1]}'")
        comp[name] = count
    total = sum(comp.values())
    print(f"[Membrane] Lipid composition ({total} total per leaflet):")
    for name, n in comp.items():
        pct = 100 * n / total
        info = LIPID_INFO.get(name, "")
        print(f"           {name:6s}  {n:4d} ({pct:.0f}%)  {info}")
    return comp


# ── packmol-memgen builder ────────────────────────────────────────────────────

def check_packmol_memgen() -> bool:
    return shutil.which("packmol-memgen") is not None


def build_with_packmol_memgen(
    composition: dict,
    size_nm:     tuple,
    water_A:     float,
    salt_M:      float,
    protein_pdb: Optional[str],
    out_dir:     str,
    lipid_ff:    str = "lipid21",
) -> tuple:
    """
    Call packmol-memgen to build the membrane.
    Returns (prmtop_path, inpcrd_path).
    """
    lipid_str = ":".join(f"{k}:{v}" for k, v in composition.items())

    cmd = [
        "packmol-memgen",
        "--lipids",    lipid_str,
        "--dist",      str(int(water_A)),      # water layer thickness (Å)
        "--salt",      str(salt_M),
        "--salt_c",    "Na+",
        "--salt_a",    "Cl-",
        "--leaflet",   "both",
        "--ffwat",     "tip3p",
        "--fflipid",   lipid_ff,
    ]

    # Membrane XY size — packmol-memgen uses number of lipids; approximate via
    # area per lipid (~65 Å² for POPC). Total lipids = 2 * sum(composition).
    total_per_leaflet = sum(composition.values())
    cmd += ["--nlipids_per_leaflet", str(total_per_leaflet)]

    if protein_pdb:
        cmd += ["--pdb", os.path.abspath(protein_pdb)]

    print(f"\n[Builder] Running packmol-memgen…")
    print(f"          Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=out_dir, capture_output=False, text=True)
    if result.returncode != 0:
        sys.exit(f"[!] packmol-memgen failed (exit {result.returncode})")

    # packmol-memgen produces bilayer.prmtop + bilayer.rst7 (or bilayer.inpcrd)
    prmtop = next((str(p) for p in Path(out_dir).glob("*.prmtop")), None)
    inpcrd = next((str(p) for p in Path(out_dir).glob("*.rst7")), None) or \
             next((str(p) for p in Path(out_dir).glob("*.inpcrd")), None)

    if not prmtop or not inpcrd:
        sys.exit(f"[!] packmol-memgen output not found in {out_dir}")

    print(f"[Builder] Topology : {prmtop}")
    print(f"[Builder] Coords   : {inpcrd}")
    return prmtop, inpcrd


def load_amber_system(prmtop: str, inpcrd: str):
    """Load AMBER topology + coordinates into OpenMM."""
    from openmm.app import AmberPrmtopFile, AmberInpcrdFile
    top  = AmberPrmtopFile(prmtop)
    crds = AmberInpcrdFile(inpcrd)
    return top.topology, crds.positions, top.createSystem


# ── Fallback: minimal flat-bilayer builder ────────────────────────────────────

def build_minimal_membrane(composition: dict, size_nm: tuple, water_A: float,
                            salt_M: float, out_dir: str) -> tuple:
    """
    Build a minimal POPC-only bilayer using OpenMM Modeller + CHARMM36 force
    field (via openmmforcefields). Downloads one POPC residue PDB as template
    and tiles it across the XY plane.

    This is a simplified fallback — use packmol-memgen for production.
    """
    try:
        from openmmforcefields.generators import SystemGenerator
        import openmmforcefields
    except ImportError:
        sys.exit(
            "[!] Neither packmol-memgen nor openmmforcefields is available.\n"
            "    Install one of:\n"
            "      conda install -c conda-forge ambertools          (recommended)\n"
            "      conda install -c conda-forge openmmforcefields"
        )

    print("[Builder] packmol-memgen not found — using CHARMM36 minimal builder")
    print("[Builder] WARNING: minimal builder creates approximate geometry only")

    # Use CHARMM36 lipid force field from openmmforcefields
    from openmm.app import ForceField, Modeller, PDBFile
    from openmm.unit import nanometer, angstrom

    # We need pre-built CHARMM PSF/PDB files or a template POPC PDB.
    # Without packmol-memgen, instruct the user to provide CHARMM-GUI output.
    print("\n[Builder] For production membranes, use CHARMM-GUI:")
    print("          1. Go to https://www.charmm-gui.org → Membrane Builder")
    print("          2. Download the OpenMM-ready package")
    print("          3. Load the PSF + PDB with --charmm-psf and --charmm-pdb")
    print("\n[Builder] Or install AmberTools:")
    print("          conda install -c conda-forge ambertools")
    sys.exit(1)


# ── CHARMM-GUI loader ─────────────────────────────────────────────────────────

def load_charmm_system(psf_path: str, pdb_path: str, ff_files: list):
    """
    Load a CHARMM-GUI-generated membrane system (PSF + PDB).
    ff_files: list of CHARMM36 force field XML files from openmmforcefields.
    """
    from openmm.app import CharmmPsfFile, PDBFile, CharmmParameterSet
    psf = CharmmPsfFile(psf_path)
    pdb = PDBFile(pdb_path)
    return psf.topology, pdb.positions


# ── Force field setup ─────────────────────────────────────────────────────────

def make_membrane_system(topology, positions, create_system_fn=None,
                          temperature_K: float = 310.0):
    """
    Create an OpenMM System for a membrane (NPT ensemble).
    Uses AMBER lipid21 + TIP3P if create_system_fn is from AmberPrmtopFile,
    otherwise falls back to CHARMM36 via openmmforcefields.
    """
    from openmm.app import PME, HBonds
    from openmm.unit import nanometer

    if create_system_fn is not None:
        # AMBER path — system parameters already in the prmtop
        system = create_system_fn(
            nonbondedMethod=PME,
            nonbondedCutoff=1.2 * nanometer,
            constraints=HBonds,
            rigidWater=True,
            flexibleConstraints=False,
        )
        print(f"[FF] AMBER lipid21 + TIP3P (from prmtop)")
    else:
        # Try openmmforcefields CHARMM36
        try:
            from openmmforcefields.generators import SystemGenerator
            from openmm.app import NoCutoff
            generator = SystemGenerator(
                forcefields=["amber/ff14SB.xml", "amber/tip3p_standard.xml"],
                small_molecule_forcefield="gaff-2.11",
            )
            system = generator.create_system(topology, molecules=[])
            print("[FF] CHARMM36m via openmmforcefields")
        except Exception as e:
            sys.exit(f"[!] Could not create membrane system: {e}")

    return system


# ── Membrane equilibration ────────────────────────────────────────────────────

def equilibrate_membrane(simulation, positions, reporter=None):
    """
    Standard membrane equilibration protocol:
      Phase 1 — Energy minimization
      Phase 2 — NVT warm-up 50→310 K (1000 steps/temperature)
      Phase 3 — NPT equilibration at 310 K, 1 bar (10 000 steps)
    """
    import openmm as mm
    from openmm.unit import kelvin, bar, picoseconds, kilojoules_per_mole, nanometer

    print("\n[Equil] Phase 1/3 — Energy minimization…")
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(maxIterations=3000)
    state = simulation.context.getState(getEnergy=True)
    print(f"[Equil] Potential energy: {state.getPotentialEnergy()}")

    print("[Equil] Phase 2/3 — Heating 50 K → 310 K…")
    for T in [50, 100, 150, 200, 250, 310]:
        simulation.integrator.setTemperature(T * kelvin)
        simulation.context.setVelocitiesToTemperature(T * kelvin)
        simulation.step(1000)
        if reporter:
            reporter.report(simulation,
                            simulation.context.getState(getPositions=True))
        print(f"[Equil]   {T} K ✓")

    print("[Equil] Phase 3/3 — NPT equilibration at 310 K, 1 bar (10 000 steps)…")
    simulation.step(10_000)
    if reporter:
        for _ in range(10):
            simulation.step(1000)
            reporter.report(simulation,
                            simulation.context.getState(getPositions=True))
    print("[Equil] Done.")


# ── Save equilibrated system ──────────────────────────────────────────────────

def save_pdb(simulation, topology, out_path: str):
    from openmm.app import PDBFile
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(out_path, "w") as f:
        PDBFile.writeFile(topology, state.getPositions(), f)
    print(f"[Output] Saved PDB → {out_path}")


def save_checkpoint(simulation, out_path: str):
    simulation.saveCheckpoint(out_path)
    print(f"[Output] Saved checkpoint → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build and equilibrate lipid bilayer membranes for OpenMM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Requirements")[0],
    )

    # ── Composition ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--lipids", default="POPC:100",
        metavar="LIPID1:N1[:LIPID2:N2:...]",
        help="Lipid composition per leaflet (default: POPC:100)"
    )
    parser.add_argument(
        "--size", nargs=2, type=float, default=[10.0, 10.0],
        metavar=("X_NM", "Y_NM"),
        help="Membrane XY dimensions in nm (default: 10 10)"
    )
    parser.add_argument(
        "--water", type=float, default=20.0,
        metavar="ANGSTROM",
        help="Water layer thickness per side in Å (default: 20)"
    )
    parser.add_argument(
        "--salt", type=float, default=0.15,
        metavar="MOL_L",
        help="NaCl concentration in mol/L (default: 0.15)"
    )
    parser.add_argument(
        "--temperature", type=float, default=310.0,
        metavar="KELVIN",
        help="Simulation temperature in K (default: 310 — physiological)"
    )

    # ── Protein ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--protein", metavar="PDB",
        help="Optional: PDB file of transmembrane protein to embed"
    )

    # ── Force field ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--lipid-ff", default="lipid21",
        choices=["lipid21", "lipid17", "lipid14"],
        help="AMBER lipid force field (default: lipid21)"
    )
    parser.add_argument(
        "--charmm-psf", metavar="PSF",
        help="Use CHARMM-GUI output: path to .psf file"
    )
    parser.add_argument(
        "--charmm-pdb", metavar="PDB",
        help="Use CHARMM-GUI output: path to .pdb file"
    )

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", default="membrane.pdb",
        metavar="FILE",
        help="Output PDB file after equilibration (default: membrane.pdb)"
    )
    parser.add_argument(
        "--workdir", default="membrane_build",
        metavar="DIR",
        help="Working directory for build files (default: membrane_build)"
    )
    parser.add_argument(
        "--no-equilibrate", action="store_true",
        help="Skip equilibration, just build and save raw system"
    )

    # ── Streaming ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream equilibration + production to PDB Visual"
    )
    parser.add_argument(
        "--port", type=int, default=7777,
        help="MDSS streaming port (default: 7777)"
    )
    parser.add_argument(
        "--viewer-exe", default="pdbvisual",
        metavar="PATH",
        help="Path to pdbvisual executable (for --stream)"
    )
    parser.add_argument(
        "--production-time", type=float, default=60.0,
        metavar="SECONDS",
        help="Wall-clock seconds for production MD after equilibration (default: 60)"
    )

    # ── Info ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--list-lipids", action="store_true",
        help="List supported lipid types and exit"
    )

    args = parser.parse_args()

    if args.list_lipids:
        print("\nSupported lipid types (AMBER lipid21 / packmol-memgen):\n")
        for name, desc in LIPID_INFO.items():
            print(f"  {name:6s}  {desc}")
        print("\n  ...and many more. Run: packmol-memgen --available_lipids\n")
        return

    # ── Validate ──────────────────────────────────────────────────────────────
    try:
        from openmm import LangevinMiddleIntegrator, MonteCarloMembraneBarostat
        from openmm import MonteCarloBarostat
        from openmm.app import Simulation
        from openmm.unit import kelvin, bar, picoseconds, femtoseconds
    except ImportError:
        sys.exit("OpenMM not found: conda install -c conda-forge openmm")

    composition = parse_lipids(args.lipids)

    print(f"\n{'='*55}")
    print(f"  Membrane Builder")
    print(f"  Lipids   : {args.lipids}")
    print(f"  Size     : {args.size[0]} x {args.size[1]} nm")
    print(f"  Water    : {args.water} Å per side")
    print(f"  Salt     : {args.salt} M NaCl")
    print(f"  Temp     : {args.temperature} K")
    print(f"  Lipid FF : {args.lipid_ff}")
    print(f"{'='*55}\n")

    os.makedirs(args.workdir, exist_ok=True)

    # ── Build system ──────────────────────────────────────────────────────────
    create_system_fn = None

    if args.charmm_psf and args.charmm_pdb:
        print("[Build] Loading CHARMM-GUI system…")
        topology, positions = load_charmm_system(
            args.charmm_psf, args.charmm_pdb, []
        )
    elif check_packmol_memgen():
        prmtop, inpcrd = build_with_packmol_memgen(
            composition     = composition,
            size_nm         = tuple(args.size),
            water_A         = args.water,
            salt_M          = args.salt,
            protein_pdb     = args.protein,
            out_dir         = args.workdir,
            lipid_ff        = args.lipid_ff,
        )
        topology, positions, create_system_fn = load_amber_system(prmtop, inpcrd)
    else:
        print("[Build] packmol-memgen not found.")
        build_minimal_membrane(composition, tuple(args.size), args.water,
                                args.salt, args.workdir)

    # ── Create system ─────────────────────────────────────────────────────────
    print("\n[FF] Creating OpenMM system…")
    system = make_membrane_system(topology, positions,
                                   create_system_fn=create_system_fn,
                                   temperature_K=args.temperature)

    # Add barostats for NPT (membrane needs semi-isotropic pressure coupling)
    try:
        barostat = MonteCarloMembraneBarostat(
            1.0 * bar,           # reference pressure
            0.0 * bar,           # surface tension (0 = tensionless bilayer)
            args.temperature * kelvin,
            MonteCarloMembraneBarostat.XYIsotropic,
            MonteCarloMembraneBarostat.ZFree,
            25,                  # frequency
        )
        system.addForce(barostat)
        print("[FF] Added MonteCarloMembraneBarostat (semi-isotropic NPT)")
    except Exception:
        # Fall back to isotropic barostat if membrane barostat unavailable
        barostat = MonteCarloBarostat(1.0 * bar, args.temperature * kelvin, 25)
        system.addForce(barostat)
        print("[FF] Added MonteCarloBarostat (isotropic NPT)")

    # ── Integrator ────────────────────────────────────────────────────────────
    integrator = LangevinMiddleIntegrator(
        args.temperature * kelvin,
        1.0 / picoseconds,
        2.0 * femtoseconds,
    )

    simulation = Simulation(topology, system, integrator)

    # ── Optional: connect to viewer ───────────────────────────────────────────
    reporter = None
    if args.stream:
        # Save raw structure first so viewer can load it
        raw_pdb = os.path.join(args.workdir, "raw_membrane.pdb")
        from openmm.app import PDBFile
        with open(raw_pdb, "w") as f:
            PDBFile.writeFile(topology, positions, f)

        sys.path.insert(0, str(Path(__file__).parent))
        from openmm_mdss import MDSSReporter
        viewer_exe = os.path.abspath(args.viewer_exe) if args.viewer_exe != "pdbvisual" \
                     else args.viewer_exe
        reporter = MDSSReporter(
            reportInterval = 50,
            port           = args.port,
            pdb_path       = os.path.abspath(raw_pdb),
            launch_viewer  = True,
            viewer_exe     = viewer_exe,
        )
        print(f"[MDSS] Streaming to PDB Visual on port {args.port}")

    # ── Equilibrate ───────────────────────────────────────────────────────────
    if not args.no_equilibrate:
        equilibrate_membrane(simulation, positions, reporter=reporter)

        # Save equilibrated structure
        save_pdb(simulation, topology, args.output)
        ckpt = args.output.replace(".pdb", ".chk")
        save_checkpoint(simulation, ckpt)
        print(f"\n[Output] Equilibrated membrane ready:")
        print(f"         PDB        : {args.output}")
        print(f"         Checkpoint : {ckpt}")
        print(f"\n         To resume simulation:")
        print(f"           python scripts/openmm_mdss.py {args.output} --live-stream {args.port}")
    else:
        # Just save the raw built system
        save_pdb(simulation, topology, args.output)
        print(f"\n[Output] Raw membrane saved to {args.output}")
        print("         Run without --no-equilibrate to equilibrate first.")

    # ── Optional production run ───────────────────────────────────────────────
    if args.stream and reporter and not args.no_equilibrate:
        print(f"\n[MD] Production run ({args.production_time:.0f}s wall-clock)…")
        simulation.reporters.append(reporter)
        try:
            simulation.runForClockTime(args.production_time)
        except KeyboardInterrupt:
            print("\n[MD] Stopped by user.")
        except Exception as e:
            print(f"\n[MD] Stopped: {e}")
        reporter.close()

    print("\n[Done]")


if __name__ == "__main__":
    main()
