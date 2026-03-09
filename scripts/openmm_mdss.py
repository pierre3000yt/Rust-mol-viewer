#!/usr/bin/env python3
"""
OpenMM MDSS Reporter — streams MD frames to PDB Visual in real time.

Usage
-----
1. Start the viewer with the same PDB file:
       mol-app.exe --live-stream 7777 protein.pdb

2. Run your simulation using MDSSReporter:
       python simulate.py protein.pdb

Protocol: MDSS v1 (TCP, little-endian)
  Handshake out : b"MDSS" + u32 version=1
  Handshake in  : b"MDSS" + u32 version=1 + u32 ok=1
  Per frame     : u32 frame_num + u32 n_atoms + n_atoms*3*f32 + u32 checksum
  End sentinel  : u32 0 + u32 0
"""

import socket
import struct
import subprocess
import sys
import os
import time
from typing import Optional

import numpy as np

# ── Protocol constants ───────────────────────────────────────────────────────
_MAGIC   = b"MDSS"
_VERSION = 1
_NM_TO_ANGSTROM = 10.0   # OpenMM uses nanometers; viewer uses Ångströms


class MDSSReporter:
    """
    OpenMM Reporter that streams atom positions to PDB Visual via MDSS protocol.

    Parameters
    ----------
    reportInterval : int
        Report every N integration steps.
    host : str
        Hostname where PDB Visual is listening (default: localhost).
    port : int
        TCP port (must match --live-stream <port> in the viewer).
    pdb_path : str | None
        If given and launch_viewer=True, the viewer is started automatically
        pointing at this PDB file.
    launch_viewer : bool
        If True, launch the viewer executable as a subprocess before connecting.
    viewer_exe : str
        Path to the mol-app executable (used only when launch_viewer=True).
    connect_retries : int
        How many times to retry the TCP connection (viewer may take a moment to start).
    """

    def __init__(
        self,
        reportInterval: int,
        host: str = "127.0.0.1",
        port: int = 7777,
        pdb_path: Optional[str] = None,
        launch_viewer: bool = False,
        viewer_exe: str = "mol-app",
        connect_retries: int = 20,
    ):
        self._interval    = reportInterval
        self._host        = host
        self._port        = port
        self._pdb_path    = pdb_path
        self._frame_num   = 0
        self._sock        = None
        self._proc        = None  # viewer subprocess

        if launch_viewer:
            if pdb_path is None:
                raise ValueError("pdb_path is required when launch_viewer=True")
            self._launch_viewer(viewer_exe, pdb_path, port)

        self._connect(connect_retries)

    # ── Viewer auto-launch ───────────────────────────────────────────────────

    def _launch_viewer(self, viewer_exe: str, pdb_path: str, port: int):
        # Resolve relative paths to absolute so subprocess can find them
        viewer_exe = os.path.abspath(viewer_exe)
        pdb_path   = os.path.abspath(pdb_path)
        cmd = [viewer_exe, "--live-stream", str(port), pdb_path]
        print(f"[MDSS] Launching viewer: {' '.join(cmd)}")
        # Launch detached so it doesn't block
        self._proc = subprocess.Popen(
            cmd,
            creationflags=subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0,
        )
        time.sleep(2.0)  # give viewer time to bind the port

    # ── TCP connection ───────────────────────────────────────────────────────

    def _connect(self, retries: int):
        last_err = None
        for attempt in range(retries):
            try:
                s = socket.create_connection((self._host, self._port), timeout=5)
                # Handshake
                s.sendall(_MAGIC + struct.pack("<I", _VERSION))
                resp = s.recv(12)
                if len(resp) < 12 or resp[:4] != _MAGIC:
                    raise ConnectionError(f"Bad handshake response: {resp!r}")
                ok = struct.unpack_from("<I", resp, 8)[0]
                if ok != 1:
                    raise ConnectionError(f"Viewer rejected connection (ok={ok})")
                self._sock = s
                print(f"[MDSS] Connected to PDB Visual at {self._host}:{self._port}")
                return
            except (ConnectionRefusedError, OSError) as e:
                last_err = e
                if attempt < retries - 1:
                    print(f"[MDSS] Connection attempt {attempt+1}/{retries} failed, retrying…")
                    time.sleep(0.5)
        raise ConnectionError(
            f"Could not connect to PDB Visual at {self._host}:{self._port} "
            f"after {retries} attempts: {last_err}"
        )

    # ── OpenMM Reporter interface ────────────────────────────────────────────

    def describeNextReport(self, simulation):
        """Tell OpenMM when we want the next report."""
        steps = self._interval - simulation.currentStep % self._interval
        # (steps, positions, velocities, forces, energies, wrap)
        return (steps, True, False, False, False, False)

    def report(self, simulation, state):
        """Called by OpenMM every reportInterval steps."""
        if self._sock is None:
            return

        # Positions come in nanometers → convert to Ångströms
        pos_nm  = state.getPositions(asNumpy=True)._value   # numpy (N, 3) float64
        coords  = (pos_nm * _NM_TO_ANGSTROM).astype(np.float32)  # (N, 3) float32

        n_atoms = coords.shape[0]
        data    = coords.flatten().tobytes()                 # interleaved x,y,z per atom

        checksum = int(np.frombuffer(data, dtype=np.uint8).sum()) & 0xFFFFFFFF

        packet = (
            struct.pack("<II", self._frame_num, n_atoms)
            + data
            + struct.pack("<I", checksum)
        )

        try:
            self._sock.sendall(packet)
        except OSError as e:
            print(f"[MDSS] Send error on frame {self._frame_num}: {e}")
            self._sock = None

        self._frame_num += 1

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def close(self):
        """Send end-of-stream sentinel and close the connection."""
        if self._sock:
            try:
                self._sock.sendall(struct.pack("<II", 0, 0))
            except OSError:
                pass
            self._sock.close()
            self._sock = None
        print(f"[MDSS] Connection closed. Sent {self._frame_num} frames.")

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Standard residue sets ────────────────────────────────────────────────────

_PROTEIN_RESIDUES = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","HIE","HID","HIP",
    "ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "ACE","NME","NHE",                        # caps
    "NALA","NGLU","CGLY","CALA",              # terminal variants
}
_NUCLEIC_RESIDUES = {
    "DA","DT","DG","DC","DU",                 # DNA
    "A","T","G","C","U",                      # RNA
    "RA","RT","RG","RC","RU",
}
_WATER_RESIDUES = {"HOH","WAT","TIP","TIP3","SOL"}
_ION_RESIDUES   = {"NA","CL","K","MG","CA","ZN","FE","MN","CU"}


# ── Structure audit ──────────────────────────────────────────────────────────

def audit_structure(topology, positions):
    """
    Inspect topology before simulation. Returns (issues, warnings) lists.
    Issues are blocking; warnings are informational.
    """
    import numpy as np
    from openmm.unit import nanometer

    issues   = []
    warnings = []

    residue_names = [r.name.upper() for r in topology.residues()]
    n_atoms = topology.getNumAtoms()
    n_res   = topology.getNumResidues()

    # — Unknown residues (not in any known set) ——————————————————————————————
    known = _PROTEIN_RESIDUES | _NUCLEIC_RESIDUES | _WATER_RESIDUES | _ION_RESIDUES
    unknown = [n for n in set(residue_names) if n not in known]
    if unknown:
        warnings.append(
            f"Unknown residues (may need extra force field): {', '.join(sorted(unknown))}"
        )

    # — NaN / Inf coordinates ————————————————————————————————————————————————
    pos_nm = np.array([[v.x, v.y, v.z] for v in positions.in_units_of(nanometer)._value
                       if hasattr(v, 'x')], dtype=float) \
             if hasattr(positions, 'in_units_of') else \
             np.array([[p.x, p.y, p.z] for p in positions])
    if np.any(~np.isfinite(pos_nm)):
        issues.append("NaN/Inf coordinates found in input structure!")

    # — Severe clashes (atoms < 0.5 Å apart) ————————————————————————————————
    pos_A = pos_nm * 10.0
    if len(pos_A) < 5000:   # skip for very large structures (too slow)
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None
    else:
        cKDTree = None
    if len(pos_A) < 5000 and cKDTree is not None:
        tree = cKDTree(pos_A)
        pairs = tree.query_pairs(r=0.5)
        if pairs:
            warnings.append(
                f"{len(pairs)} severe atom clashes (<0.5 Å) — "
                "minimization may struggle"
            )

    # — Chain breaks (Cα–Cα > 5 Å in same chain) ————————————————————————————
    chains_with_breaks = []
    for chain in topology.chains():
        ca_positions = []
        for res in chain.residues():
            for atom in res.atoms():
                if atom.name == "CA":
                    idx = atom.index
                    ca_positions.append(pos_A[idx])
        for i in range(1, len(ca_positions)):
            d = np.linalg.norm(ca_positions[i] - ca_positions[i-1])
            if d > 5.0:
                chains_with_breaks.append(chain.id)
                break
    if chains_with_breaks:
        warnings.append(
            f"Chain break(s) detected in chain(s): {', '.join(chains_with_breaks)}"
        )

    print(f"[Audit] {n_atoms} atoms | {n_res} residues")
    for w in warnings:
        print(f"[Audit] WARNING: {w}")
    for e in issues:
        print(f"[Audit] ERROR:   {e}")
    if not issues and not warnings:
        print("[Audit] Structure looks clean.")

    return issues, warnings


# ── Force field selection ────────────────────────────────────────────────────

def select_forcefield(topology):
    """
    Choose the best available force field components for this topology.
    Returns a ForceField instance and a label string.
    """
    from openmm.app import ForceField

    residue_names = {r.name.upper() for r in topology.residues()}
    has_protein  = bool(residue_names & _PROTEIN_RESIDUES)
    has_nucleic  = bool(residue_names & _NUCLEIC_RESIDUES)
    has_water    = bool(residue_names & _WATER_RESIDUES)
    has_unknown  = bool(residue_names - (_PROTEIN_RESIDUES | _NUCLEIC_RESIDUES
                                        | _WATER_RESIDUES | _ION_RESIDUES))

    ff_files = ["amber14-all.xml"]   # covers protein + nucleic acids

    if has_water:
        ff_files.append("amber14/tip3pfb.xml")
        label = "AMBER14 + TIP3P-FB (explicit solvent)"
    else:
        ff_files.append("implicit/gbn2.xml")
        label = "AMBER14 + GBn2 (implicit solvent)"

    if has_unknown:
        # Try to load GAFF2 for small molecules
        try:
            ff = ForceField(*ff_files, "gaff2.xml")
            label += " + GAFF2"
            print(f"[FF] Force field: {label}")
            return ff, label
        except Exception:
            pass

    ff = ForceField(*ff_files)
    print(f"[FF] Force field: {label}")
    return ff, label


# ── System creation with fallbacks ──────────────────────────────────────────

def create_system(ff, topology, has_water: bool):
    """
    Try to create the OpenMM system with progressively relaxed settings
    until one succeeds.
    """
    from openmm.app import HBonds, PME, CutoffNonPeriodic
    from openmm.unit import nanometer

    # ignoreExternalBonds removes bonded terms → wrong forces → NaN
    # Only use as absolute last resort; flag it clearly
    attempts = [
        dict(nonbondedMethod=PME if has_water else CutoffNonPeriodic,
             constraints=HBonds),
        dict(nonbondedMethod=CutoffNonPeriodic,
             constraints=HBonds),
        dict(nonbondedMethod=CutoffNonPeriodic,
             constraints=None),
        dict(nonbondedMethod=CutoffNonPeriodic,
             constraints=None, ignoreExternalBonds=True),
    ]

    last_err = None
    used_ignore_external = False
    for i, kwargs in enumerate(attempts):
        try:
            system = ff.createSystem(topology, **kwargs)
            if i > 0:
                print(f"[FF] System created on attempt {i+1} with: {kwargs}")
            if kwargs.get("ignoreExternalBonds"):
                used_ignore_external = True
                print("[FF] WARNING: ignoreExternalBonds=True — some bonded terms "
                      "are missing. Timestep will be reduced to 0.5 fs.")
            return system, used_ignore_external
        except Exception as e:
            last_err = e
            print(f"[FF] Attempt {i+1} failed: {e}")

    raise RuntimeError(
        f"Could not create system after {len(attempts)} attempts.\n"
        f"Last error: {last_err}\n"
        "The structure may contain unsupported residues or ligands.\n"
        "Try removing non-standard residues with PDBFixer manually."
    )


# ── Chain break splitter ─────────────────────────────────────────────────────

def split_chain_breaks(pdb_path: str, out_path: str, ca_break_threshold: float = 5.0) -> int:
    """
    Detect backbone breaks (Cα–Cα > threshold Å, or duplicate residue numbers)
    and assign a new chain ID to each continuous segment.

    Maps at RESIDUE level — all atoms of a residue get the same chain ID.
    Inserts TER records between chain changes so PDB readers see clean termini.

    Returns the number of splits made.
    """
    import string
    chain_pool = list(string.ascii_uppercase + string.ascii_lowercase)

    with open(pdb_path) as f:
        lines = f.readlines()

    # ── Pass 1: collect one Cα per residue, in order ─────────────────────────
    res_order: list = []          # (orig_chain, resnum, icode)
    res_ca:    dict = {}          # key → (x, y, z)
    seen:      set  = set()

    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        orig_chain = line[21]
        resnum     = int(line[22:26])
        icode      = line[26]
        key = (orig_chain, resnum, icode)
        if key in seen:
            continue
        seen.add(key)
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        res_order.append(key)
        res_ca[key] = (x, y, z)

    if not res_order:
        import shutil; shutil.copy(pdb_path, out_path)
        return 0

    # ── Pass 2: assign new chain ID per residue ───────────────────────────────
    res_new_chain: dict = {}
    pool_idx      = 0
    current_id    = res_order[0][0]   # start with original chain ID
    n_splits      = 0

    for k, key in enumerate(res_order):
        if k == 0:
            res_new_chain[key] = current_id
            continue

        prev = res_order[k - 1]

        # Different original chain → keep original ID (don't merge separate chains)
        if prev[0] != key[0]:
            current_id = key[0]
            res_new_chain[key] = current_id
            continue

        # Same original chain — check for break
        px, py, pz = res_ca[prev]
        cx, cy, cz = res_ca[key]
        dist = ((cx-px)**2 + (cy-py)**2 + (cz-pz)**2) ** 0.5
        same_resnum = (prev[1] == key[1])

        if same_resnum or dist > ca_break_threshold:
            pool_idx += 1
            # Pick a fresh chain ID not already used
            while pool_idx < len(chain_pool) and chain_pool[pool_idx] in {
                    v for v in res_new_chain.values()}:
                pool_idx += 1
            current_id = chain_pool[pool_idx] if pool_idx < len(chain_pool) else "Z"
            n_splits += 1

        res_new_chain[key] = current_id

    if n_splits == 0:
        import shutil; shutil.copy(pdb_path, out_path)
        return 0

    # ── Pass 3: rewrite PDB with patched chain IDs + TER records ─────────────
    new_lines     = []
    prev_chain_id = None

    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            orig_chain = line[21]
            resnum     = int(line[22:26])
            icode      = line[26]
            key        = (orig_chain, resnum, icode)
            new_chain  = res_new_chain.get(key, orig_chain)

            if prev_chain_id is not None and new_chain != prev_chain_id:
                new_lines.append("TER\n")

            line = line[:21] + new_chain + line[22:]
            prev_chain_id = new_chain

        elif line.startswith("TER"):
            prev_chain_id = None   # reset; the PDB already marks a chain end

        new_lines.append(line)

    with open(out_path, "w") as f:
        f.writelines(new_lines)

    return n_splits


# ── Structure preparation ────────────────────────────────────────────────────

def prepare_model(pdb_path: str):
    """
    Full preparation pipeline:
      0. Split any chain breaks into separate chain IDs (fixes duplicate-number chains)
      1. PDBFixer: missing atoms, non-standard → standard, remove heterogens, add H
      2. Save fixed PDB (topology must match what the viewer loads)
    Returns (topology, positions, fixed_pdb_path)
    """
    from openmm.app import PDBFile, Modeller

    # ── Step 0: split chains at backbone breaks ──────────────────────────────
    split_path = pdb_path.replace(".pdb", "_split.pdb")
    n_splits = split_chain_breaks(pdb_path, split_path)
    if n_splits:
        print(f"[Prep] Split {n_splits} chain break(s) into separate chains → {split_path}")
        pdb_path = split_path

    try:
        from pdbfixer import PDBFixer
        print("[Prep] Running PDBFixer…")
        fixer = PDBFixer(filename=pdb_path)

        fixer.findMissingResidues()
        # Don't fill internal gaps — each chain now has clean termini
        # (chain breaks were already split into separate chains by split_chain_breaks)
        fixer.missingResidues = {}

        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        # Do NOT call fixer.addMissingHydrogens() here — PDBFixer adds ACE/NME
        # terminal caps that conflict with AMBER14's own terminal templates
        # (NGLU, CGLY, etc.). Let Modeller.addHydrogens() use the FF templates.

        # Remove any ACE/NME caps PDBFixer may have already inserted
        caps = [r for r in fixer.topology.residues() if r.name in ("ACE", "NME", "NHE")]
        if caps:
            print(f"[Prep] Removing {len(caps)} terminal cap(s) added by PDBFixer "
                  "(AMBER14 handles termini natively)")
            mod_caps = Modeller(fixer.topology, fixer.positions)
            mod_caps.delete(caps)
            fixer_topology, fixer_positions = mod_caps.topology, mod_caps.positions
        else:
            fixer_topology, fixer_positions = fixer.topology, fixer.positions

        # Add hydrogens using AMBER14 templates — correct terminal variants
        ff_for_h, _ = select_forcefield(fixer_topology)
        mod = Modeller(fixer_topology, fixer_positions)
        try:
            mod.addHydrogens(ff_for_h, pH=7.0)
            topology, positions = mod.topology, mod.positions
        except Exception as e:
            print(f"[Prep] addHydrogens failed ({e}), using heavy-atom structure")
            topology, positions = fixer_topology, fixer_positions

    except ImportError:
        print("[Prep] PDBFixer not installed — using basic Modeller")
        print("       conda install -c conda-forge pdbfixer  (recommended)")
        # Select force field just for addHydrogens
        ff, _ = select_forcefield(PDBFile(pdb_path).topology)
        pdb = PDBFile(pdb_path)
        mod = Modeller(pdb.topology, pdb.positions)
        mod.addHydrogens(ff, pH=7.0)
        topology, positions = mod.topology, mod.positions

    fixed_path = pdb_path.replace(".pdb", "_fixed.pdb")
    with open(fixed_path, "w") as f:
        PDBFile.writeFile(topology, positions, f)
    print(f"[Prep] Saved prepared model → {fixed_path}")

    return topology, positions, fixed_path


# ── Equilibration ────────────────────────────────────────────────────────────

def equilibrate(simulation, positions):
    """
    Simple, safe equilibration:
      1. Set positions and minimize (remove clashes)
      2. Assign velocities at 300 K
      3. Short NVT warm-up (not streamed to viewer)
    """
    from openmm.unit import kelvin

    print("[Equil] Energy minimization…")
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(maxIterations=2000)
    state = simulation.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy()
    print(f"[Equil] Potential energy: {pe}")

    simulation.context.setVelocitiesToTemperature(300 * kelvin)

    print("[Equil] Short warm-up (500 steps)…")
    simulation.step(500)
    print("[Equil] Ready for production MD")


# ── Main simulation pipeline ─────────────────────────────────────────────────

def run_example(pdb_path: str, port: int = 7777, launch_viewer: bool = False,
                viewer_exe: str = "mol-app"):
    """
    Full pipeline:
      prepare → audit → select FF → create system → equilibrate → production
    """
    try:
        import openmm as mm
        from openmm import LangevinMiddleIntegrator
        from openmm.app import Simulation
        from openmm.unit import kelvin, picosecond, picoseconds
    except ImportError:
        sys.exit("OpenMM not found. Install with:  conda install -c conda-forge openmm")

    print(f"\n{'='*55}")
    print(f"  PDB Visual — OpenMM simulation pipeline")
    print(f"  Input: {pdb_path}")
    print(f"{'='*55}\n")

    # ── 1. Prepare ───────────────────────────────────────────────────────────
    topology, positions, viewer_pdb = prepare_model(pdb_path)

    # ── 2. Audit ─────────────────────────────────────────────────────────────
    issues, warnings = audit_structure(topology, positions)
    if issues:
        print("\n[!] Blocking issues found — cannot simulate safely.")
        for e in issues:
            print(f"    {e}")
        sys.exit(1)

    # ── 3. Force field ───────────────────────────────────────────────────────
    has_water = any(r.name.upper() in _WATER_RESIDUES for r in topology.residues())
    ff, ff_label = select_forcefield(topology)

    # ── 4. System ────────────────────────────────────────────────────────────
    print(f"[FF] Creating system…")
    system, used_ignore_external = create_system(ff, topology, has_water)

    # ── 5. Integrator & Simulation ───────────────────────────────────────────
    # Use 0.5 fs if ignoreExternalBonds was needed (missing bonded terms = stiff)
    dt = 0.0005 if used_ignore_external else 0.002
    integrator = LangevinMiddleIntegrator(300 * kelvin, 1 / picosecond, dt * picoseconds)
    print(f"[MD] Timestep: {dt*1000:.1f} fs")
    simulation = Simulation(topology, system, integrator)

    # ── 6. Connect viewer ────────────────────────────────────────────────────
    print(f"\n[MDSS] Starting viewer and connecting…")
    print(f"       Viewer PDB: {viewer_pdb}")

    with MDSSReporter(
        reportInterval=10,
        port=port,
        pdb_path=viewer_pdb,
        launch_viewer=launch_viewer,
        viewer_exe=viewer_exe,
    ) as reporter:

        # ── 7. Equilibrate ───────────────────────────────────────────────────
        print()
        equilibrate(simulation, positions)

        # ── 8. Production ────────────────────────────────────────────────────
        simulation.reporters.append(reporter)
        print(f"\n[MD] Production run — streaming to PDB Visual (Ctrl+C to stop)…")
        try:
            simulation.runForClockTime(3600)  # up to 1 hour; Ctrl+C to stop early
        except KeyboardInterrupt:
            print("\n[MD] Stopped by user.")
        except Exception as e:
            print(f"\n[MD] Simulation error: {e}")
            print("[MD] If NaN: structure had unresolved clashes or bad termini.")

    print(f"\n[MD] Done. Sent {reporter._frame_num} frames total.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream OpenMM simulation to PDB Visual via MDSS"
    )
    parser.add_argument("pdb", help="PDB file to simulate (same file passed to the viewer)")
    parser.add_argument("--port", type=int, default=7777, help="MDSS port (default: 7777)")
    parser.add_argument(
        "--launch-viewer", action="store_true",
        help="Auto-launch mol-app.exe before connecting"
    )
    parser.add_argument(
        "--viewer-exe", default="mol-app",
        help="Path to mol-app executable (used with --launch-viewer)"
    )
    args = parser.parse_args()

    run_example(args.pdb, port=args.port, launch_viewer=args.launch_viewer,
                viewer_exe=args.viewer_exe)
