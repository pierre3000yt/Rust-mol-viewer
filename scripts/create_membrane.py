#!/usr/bin/env python3
"""
create_membrane.py — Build lipid bilayer membranes for OpenMM

Uses a built-in coarse-grained (5-bead) lipid force field — no extra
packages required beyond openmm + numpy.

For production all-atom simulations, use CHARMM-GUI output (see --charmm-psf).

Requirements
------------
    conda install -c conda-forge openmm numpy   (already installed)

Quick start
-----------
    # POPC bilayer streamed to the viewer
    python scripts/create_membrane.py --lipids POPC:100 --stream \\
        --viewer-exe target/release/pdbvisual.exe

    # Mixed bilayer (plasma membrane-like)
    python scripts/create_membrane.py \\
        --lipids POPC:40:POPE:30:CHOL:20:POPS:10 \\
        --size 100 100 --salt 0.15 --stream \\
        --viewer-exe target/release/pdbvisual.exe

    # Load CHARMM-GUI all-atom system (production quality)
    python scripts/create_membrane.py \\
        --charmm-psf step5_input.psf --charmm-pdb step5_input.pdb \\
        --stream --viewer-exe target/release/pdbvisual.exe

    # Build only, save PDB, simulate later
    python scripts/create_membrane.py --lipids DPPC:100 --output dppc.pdb

Supported lipids
----------------
    POPC POPE POPS POPG POPA  DPPC DPPE DPPS DPPG DPPA
    DOPC DOPE DOPS DOPG DOPA  DLPC DLPE DLPS DLPG
    DSPC DSPE DSPS DSPG DSPA  CHOL
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


# ── Built-in CG force field ───────────────────────────────────────────────────
# 5-bead model per lipid: P (headgroup) – O (glycerol) – C1 – C2 – CT (tail end)
# Parameters inspired by MARTINI 2 but adapted for OpenMM implicit solvent.
# CHOL uses a 4-bead model (no glycerol bead).

def _make_residue_xml(name: str, n_beads: int = 5) -> str:
    if n_beads == 5:
        return f"""
    <Residue name="{name}">
      <Atom name="P"   type="LP"   charge="-0.40" />
      <Atom name="O1"  type="LO"   charge="+0.20" />
      <Atom name="C1"  type="LC"   charge="+0.20" />
      <Atom name="C2"  type="LC"   charge="0.00"  />
      <Atom name="CT"  type="LCT"  charge="0.00"  />
      <Bond from="0" to="1"/>
      <Bond from="1" to="2"/>
      <Bond from="2" to="3"/>
      <Bond from="3" to="4"/>
    </Residue>"""
    else:  # 4-bead (CHOL)
        return f"""
    <Residue name="{name}">
      <Atom name="O1"  type="LO"   charge="+0.20" />
      <Atom name="C1"  type="LC"   charge="-0.10" />
      <Atom name="C2"  type="LC"   charge="-0.10" />
      <Atom name="CT"  type="LCT"  charge="0.00"  />
      <Bond from="0" to="1"/>
      <Bond from="1" to="2"/>
      <Bond from="2" to="3"/>
    </Residue>"""


_LIPID_NAMES_5 = [
    "POPC","POPE","POPS","POPG","POPA",
    "DPPC","DPPE","DPPS","DPPG","DPPA",
    "DOPC","DOPE","DOPS","DOPG","DOPA",
    "DLPC","DLPE","DLPS","DLPG",
    "DSPC","DSPE","DSPS","DSPG","DSPA",
]

_CG_FF_XML = """<ForceField>
  <AtomTypes>
    <Type name="LP"  class="LP"  element="P"  mass="94.97" />
    <Type name="LO"  class="LO"  element="O"  mass="40.00" />
    <Type name="LC"  class="LC"  element="C"  mass="56.07" />
    <Type name="LCT" class="LCT" element="C"  mass="56.07" />
    <Type name="LN"  class="LN"  element="N"  mass="14.01" />
    <Type name="OW"  class="OW"  element="O"  mass="15.999" />
    <Type name="HW"  class="HW"  element="H"  mass="1.008"  />
    <Type name="Ion-Na" class="Ion-Na" element="Na" mass="22.990" />
    <Type name="Ion-Cl" class="Ion-Cl" element="Cl" mass="35.453" />
  </AtomTypes>

  <Residues>
    {residues}
    <Residue name="HOH">
      <Atom name="O"  type="OW"     charge="-0.834" />
      <Atom name="H1" type="HW"     charge="+0.417" />
      <Atom name="H2" type="HW"     charge="+0.417" />
      <Bond from="0" to="1"/>
      <Bond from="0" to="2"/>
    </Residue>
    <Residue name="NA">
      <Atom name="NA" type="Ion-Na" charge="+1.0" />
    </Residue>
    <Residue name="CL">
      <Atom name="CL" type="Ion-Cl" charge="-1.0" />
    </Residue>
  </Residues>

  <HarmonicBondForce>
    <Bond class1="LP"  class2="LO"  length="0.37"    k="4000"/>
    <Bond class1="LO"  class2="LC"  length="0.37"    k="4000"/>
    <Bond class1="LC"  class2="LC"  length="0.37"    k="4000"/>
    <Bond class1="LC"  class2="LCT" length="0.37"    k="4000"/>
    <Bond class1="LO"  class2="LCT" length="0.37"    k="4000"/>
    <Bond class1="OW"  class2="HW"  length="0.09572" k="462750"/>
  </HarmonicBondForce>

  <HarmonicAngleForce>
    <Angle class1="LP"  class2="LO"  class3="LC"  angle="2.094" k="100"/>
    <Angle class1="LO"  class2="LC"  class3="LC"  angle="2.094" k="100"/>
    <Angle class1="LC"  class2="LC"  class3="LCT" angle="2.094" k="100"/>
    <Angle class1="LO"  class2="LC"  class3="LCT" angle="2.094" k="100"/>
    <Angle class1="HW"  class2="OW"  class3="HW"  angle="1.824" k="836"/>
  </HarmonicAngleForce>

  <NonbondedForce coulomb14scale="1.0" lj14scale="1.0">
    <UseAttributeFromResidue name="charge"/>
    <Atom type="LP"     sigma="0.470" epsilon="3.500"/>
    <Atom type="LO"     sigma="0.470" epsilon="3.500"/>
    <Atom type="LC"     sigma="0.470" epsilon="3.500"/>
    <Atom type="LCT"    sigma="0.470" epsilon="3.500"/>
    <Atom type="LN"     sigma="0.430" epsilon="3.500"/>
    <Atom type="OW"     sigma="0.31507" epsilon="0.6364"/>
    <Atom type="HW"     sigma="0.1"     epsilon="0.0"/>
    <Atom type="Ion-Na" sigma="0.2439"  epsilon="0.3658"/>
    <Atom type="Ion-Cl" sigma="0.4045"  epsilon="0.6276"/>
  </NonbondedForce>
</ForceField>
"""

def _build_cg_ff_xml() -> str:
    residues = "\n".join(
        _make_residue_xml(n, 5) for n in _LIPID_NAMES_5
    ) + "\n" + _make_residue_xml("CHOL", 4)
    return _CG_FF_XML.format(residues=residues)


def get_cg_forcefield():
    """Return an OpenMM ForceField built from the inline CG definition."""
    from openmm.app import ForceField
    xml = _build_cg_ff_xml()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(xml)
        tmp = f.name
    try:
        ff = ForceField(tmp)
    finally:
        os.unlink(tmp)
    return ff


# ── Lipid geometry ────────────────────────────────────────────────────────────

LIPID_INFO = {
    #       area per lipid (Å²)  z_head (Å from center)  n_beads
    "POPC": (68.0, 18.0, 5),
    "POPE": (60.0, 18.0, 5),
    "POPS": (60.0, 18.0, 5),
    "POPG": (65.0, 18.0, 5),
    "DPPC": (63.0, 17.0, 5),
    "DPPE": (55.0, 17.0, 5),
    "DOPC": (72.0, 19.0, 5),
    "DOPE": (65.0, 19.0, 5),
    "DSPC": (60.0, 17.0, 5),
    "DLPC": (60.0, 15.0, 5),
    "CHOL": (38.0, 14.0, 4),
}
_DEFAULT = (65.0, 18.0, 5)


# ── Composition parser ────────────────────────────────────────────────────────

def parse_lipids(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) % 2 != 0:
        sys.exit(f"[!] Bad lipid spec '{spec}'. Expected LIPID1:N1:LIPID2:N2")
    comp = {}
    for i in range(0, len(parts), 2):
        name = parts[i].upper()
        try:
            n = int(parts[i+1])
        except ValueError:
            sys.exit(f"[!] Expected integer after '{name}', got '{parts[i+1]}'")
        comp[name] = n

    total = sum(comp.values())
    descs = {
        "POPC": "PC headgroup — most common",
        "POPE": "PE headgroup — inner leaflet",
        "POPS": "PS headgroup — anionic, inner leaflet",
        "POPG": "PG headgroup — anionic (bacterial)",
        "DPPC": "PC saturated tails (Tm=41°C)",
        "DOPC": "PC unsaturated tails",
        "DOPE": "PE unsaturated tails",
        "CHOL": "Cholesterol — modulates fluidity",
    }
    print(f"[Membrane] Composition ({total} per leaflet):")
    for name, n in comp.items():
        print(f"           {name:6s} {n:4d} ({100*n//total}%)  {descs.get(name,'')}")

    unknown = [k for k in comp if k not in LIPID_INFO]
    if unknown:
        print(f"[Membrane] WARNING: unknown lipids {unknown} — using default geometry")
    return comp


# ── Bilayer builder ───────────────────────────────────────────────────────────

def build_bilayer(composition: dict, box_xy_A: tuple,
                  z_water_A: float) -> tuple:
    """
    Build a coarse-grained lipid bilayer.
    Returns (topology, positions_nm, box_vectors_nm).

    Upper leaflet: heads at +z_head, tails toward z=0
    Lower leaflet: heads at -z_head, tails toward z=0
    """
    from openmm.app import Topology, Element
    import openmm.unit as unit

    np.random.seed(42)   # reproducible jitter

    total   = sum(composition.values())
    avg_apl = sum(composition[k] * LIPID_INFO.get(k, _DEFAULT)[0]
                  for k in composition) / total
    box_area = box_xy_A[0] * box_xy_A[1]
    n_lip    = max(total, int(box_area / avg_apl))

    nx = max(1, int(np.ceil(np.sqrt(n_lip * box_xy_A[0] / box_xy_A[1]))))
    ny = max(1, int(np.ceil(n_lip / nx)))
    dx = box_xy_A[0] / nx
    dy = box_xy_A[1] / ny
    print(f"[Builder] Grid {nx}×{ny} = {nx*ny} positions/leaflet, "
          f"spacing {dx:.1f}×{dy:.1f} Å")

    # Expand lipid sequence to fill grid
    seq = []
    for name, count in composition.items():
        seq.extend([name] * count)
    while len(seq) < n_lip:
        seq = seq + seq
    seq = seq[:n_lip]
    np.random.shuffle(seq)

    topology  = Topology()
    positions = []

    bead_names = {5: ["P", "O1", "C1", "C2", "CT"],
                  4: ["O1", "C1", "C2", "CT"]}
    bead_elems = {5: ["P", "O", "C", "C", "C"],
                  4: ["O", "C", "C", "C"]}

    def add_lipid(chain, name: str, cx: float, cy: float, flip: bool):
        apl, z_head, n_beads = LIPID_INFO.get(name, _DEFAULT)
        bead_len = z_head / (n_beads - 1) if n_beads > 1 else z_head
        sign     = -1.0 if flip else 1.0

        res    = topology.addResidue(name, chain)
        names  = bead_names[n_beads]
        elems  = bead_elems[n_beads]

        atoms = []
        for i in range(n_beads):
            # head (i=0) at z_head, tail (i=n-1) at z=0
            z  = sign * (z_head - i * bead_len)
            jx = np.random.uniform(-0.4, 0.4)
            jy = np.random.uniform(-0.4, 0.4)
            atoms.append(topology.addAtom(names[i], Element.getBySymbol(elems[i]), res))
            positions.append([(cx + jx) * 0.1,
                               (cy + jy) * 0.1,
                               z          * 0.1])   # nm
        # Add linear bonds along the bead chain (required for FF template matching)
        for i in range(n_beads - 1):
            topology.addBond(atoms[i], atoms[i + 1])

    for leaflet_flip in [False, True]:
        chain = topology.addChain()
        for idx in range(n_lip):
            row = idx // nx
            col = idx %  nx
            add_lipid(chain, seq[idx],
                      (col + 0.5) * dx,
                      (row + 0.5) * dy,
                      leaflet_flip)

    max_z  = max(LIPID_INFO.get(k, _DEFAULT)[1] for k in composition)
    box_z  = 2.0 * (max_z + z_water_A)
    box_nm = np.array([
        [box_xy_A[0] * 0.1, 0.0, 0.0],
        [0.0, box_xy_A[1] * 0.1, 0.0],
        [0.0, 0.0,               box_z * 0.1],
    ])

    pos_qty = unit.Quantity(positions, unit.nanometer)
    print(f"[Builder] {topology.getNumAtoms()} beads, "
          f"box {box_xy_A[0]:.0f}×{box_xy_A[1]:.0f}×{box_z:.0f} Å")
    return topology, pos_qty, box_nm


# ── Water + ions ──────────────────────────────────────────────────────────────

def add_water_and_ions(topology, positions_nm, box_nm,
                        z_water_A: float, salt_M: float) -> tuple:
    """
    Place TIP3P water molecules and NaCl ions in the water layers.
    Water layers: |z| > z_membrane (i.e., above/below the bilayer).
    """
    from openmm.app import Topology, Element
    import openmm.unit as unit

    water_spacing_nm = 0.31   # nm between water oxygens (~TIP3P density)
    box_x = box_nm[0][0]
    box_y = box_nm[1][1]
    box_z = box_nm[2][2]

    z_layer_nm = z_water_A * 0.1       # nm
    z_bilayer  = (box_z / 2) - z_layer_nm   # nm from center to bilayer edge

    nx = max(1, int(box_x / water_spacing_nm))
    ny = max(1, int(box_y / water_spacing_nm))
    dx = box_x / nx
    dy = box_y / ny

    # Water z positions: top and bottom slabs
    z_top    = np.arange(z_bilayer + 0.15, box_z - 0.15, water_spacing_nm)
    z_bottom = np.arange(0.15, (box_z / 2) - z_bilayer - 0.15, water_spacing_nm)
    z_vals   = np.concatenate([z_top, z_bottom])

    water_positions = []
    for iz, z in enumerate(z_vals):
        for iy in range(ny):
            for ix in range(nx):
                x = (ix + 0.5 * (iz % 2)) * dx + np.random.uniform(-0.05, 0.05)
                y = (iy + 0.5 * (iz % 2)) * dy + np.random.uniform(-0.05, 0.05)
                water_positions.append((x % box_x, y % box_y, z))

    n_water = len(water_positions)

    # Ion counts from salt concentration
    avogadro = 6.022e23
    vol_L = box_x * box_y * box_z * 1e-24 * 1000   # nm³ → L
    n_ions = max(0, int(salt_M * avogadro * vol_L))
    n_na   = n_ions
    n_cl   = n_ions
    print(f"[Solv] {n_water} water molecules, {n_na} Na⁺, {n_cl} Cl⁻")

    # Steal some water positions for ions
    ion_positions = []
    if n_na + n_cl > 0 and len(water_positions) > n_na + n_cl:
        stride = len(water_positions) // (n_na + n_cl + 1)
        taken  = set()
        for k in range(n_na + n_cl):
            idx = (k + 1) * stride
            ion_positions.append(water_positions[idx])
            taken.add(idx)
        water_positions = [p for i, p in enumerate(water_positions) if i not in taken]

    # Build combined topology
    new_top  = Topology()
    new_pos  = list(positions_nm._value if hasattr(positions_nm, '_value')
                    else positions_nm)

    # Copy existing chains
    atom_map = {}
    for chain in topology.chains():
        nc = new_top.addChain()
        for res in chain.residues():
            nr = new_top.addResidue(res.name, nc)
            for atom in res.atoms():
                na = new_top.addAtom(atom.name, atom.element, nr)
                atom_map[atom.index] = na.index
    for bond in topology.bonds():
        new_top.addBond(
            list(new_top.atoms())[atom_map[bond.atom1.index]],
            list(new_top.atoms())[atom_map[bond.atom2.index]],
        )

    # Add water
    O_elem  = Element.getBySymbol("O")
    H_elem  = Element.getBySymbol("H")
    Na_elem = Element.getBySymbol("Na")
    Cl_elem = Element.getBySymbol("Cl")

    wchain = new_top.addChain()
    for x, y, z in water_positions:
        wr = new_top.addResidue("HOH", wchain)
        ow  = new_top.addAtom("O",  O_elem, wr)
        hw1 = new_top.addAtom("H1", H_elem, wr)
        hw2 = new_top.addAtom("H2", H_elem, wr)
        new_top.addBond(ow, hw1)
        new_top.addBond(ow, hw2)
        # TIP3P geometry: O at center, H at ±95.7° with O-H = 0.09572 nm
        new_pos.append([x, y, z])
        new_pos.append([x + 0.0757, y + 0.0586, z])
        new_pos.append([x - 0.0757, y + 0.0586, z])

    # Add ions
    ichain = new_top.addChain()
    for k, (x, y, z) in enumerate(ion_positions):
        if k < n_na:
            ir = new_top.addResidue("NA", ichain)
            new_top.addAtom("NA", Na_elem, ir)
        else:
            ir = new_top.addResidue("CL", ichain)
            new_top.addAtom("CL", Cl_elem, ir)
        new_pos.append([x, y, z])

    return new_top, unit.Quantity(new_pos, unit.nanometer)


# ── System creation ───────────────────────────────────────────────────────────

def create_cg_system(topology, ff, box_nm=None):
    """
    Create OpenMM System for the CG membrane.
    Uses CutoffNonPeriodic (implicit solvent) — no PBC needed for CG preview.
    """
    from openmm.app import CutoffNonPeriodic, HBonds
    from openmm.unit import nanometer

    try:
        system = ff.createSystem(
            topology,
            nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=1.5 * nanometer,
            constraints=None,
            ignoreExternalBonds=True,
        )
        print(f"[FF] CG lipid force field — {topology.getNumAtoms()} particles")
        return system
    except Exception as e:
        raise RuntimeError(f"System creation failed: {e}\n"
                           "Check that all lipid names are in the supported list.")


def create_charmm_system(psf, ff_files: list):
    """Create system from CHARMM-GUI PSF using CHARMM36 FF."""
    from openmm.app import CharmmParameterSet, PME, HBonds, ForceField
    from openmm.unit import nanometer

    # Try CharmmParameterSet with .prm/.rtf/.str files
    prm_files = [f for f in ff_files if not f.endswith(".xml")]
    if prm_files:
        try:
            params = CharmmParameterSet(*prm_files)
            system = psf.createSystem(params, nonbondedMethod=PME,
                                       nonbondedCutoff=1.2*nanometer,
                                       constraints=HBonds)
            print("[FF] CHARMM36 (parameter files)")
            return system
        except Exception as e:
            print(f"[FF] CharmmParameterSet failed: {e}")

    # Try OpenMM ForceField with XML files
    xml_files = [f for f in ff_files if f.endswith(".xml")]
    if not xml_files:
        xml_files = ["charmm36.xml", "charmm36/water.xml"]
    try:
        ff = ForceField(*xml_files)
        system = ff.createSystem(psf.topology, nonbondedMethod=PME,
                                  nonbondedCutoff=1.2*nanometer,
                                  constraints=HBonds)
        print(f"[FF] CHARMM36 (XML)")
        return system
    except Exception as e:
        raise RuntimeError(f"Cannot create CHARMM system: {e}\n"
                           "Make sure CHARMM36 XML files are available.")


# ── CHARMM-GUI loader ─────────────────────────────────────────────────────────

def load_charmm_gui(psf_path: str, pdb_path: str):
    from openmm.app import CharmmPsfFile, PDBFile
    print(f"[Build] Loading CHARMM-GUI system…")
    psf = CharmmPsfFile(psf_path)
    pdb = PDBFile(pdb_path)
    n = psf.topology.getNumAtoms()
    r = psf.topology.getNumResidues()
    print(f"[Build] {n} atoms, {r} residues")
    return psf, pdb.positions


# ── Equilibration ─────────────────────────────────────────────────────────────

def equilibrate(simulation, positions, temperature_K: float, reporter=None):
    from openmm.unit import kelvin

    print("\n[Equil] Phase 1/2 — Energy minimization…")
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(maxIterations=3000)
    E = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"[Equil] Energy: {E}")

    print(f"[Equil] Phase 2/2 — Heating to {temperature_K:.0f} K…")
    for T in [50, 100, 150, 200, 250, int(temperature_K)]:
        simulation.integrator.setTemperature(T * kelvin)
        simulation.context.setVelocitiesToTemperature(T * kelvin)
        simulation.step(500)
        if reporter:
            reporter.report(simulation,
                            simulation.context.getState(getPositions=True))
        print(f"[Equil]   {T} K ✓")
    print("[Equil] Done.")


# ── I/O ───────────────────────────────────────────────────────────────────────

def save_pdb(simulation, topology, path: str):
    from openmm.app import PDBFile
    state = simulation.context.getState(getPositions=True)
    with open(path, "w") as f:
        PDBFile.writeFile(topology, state.getPositions(), f)
    print(f"[Out] PDB → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Build lipid bilayer membranes for OpenMM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--lipids", default="POPC:100",
                   metavar="LIPID:N[:LIPID:N]",
                   help="Composition per leaflet (default: POPC:100)")
    p.add_argument("--size", nargs=2, type=float, default=[80.0, 80.0],
                   metavar=("X_A", "Y_A"),
                   help="Box XY in Ångström (default: 80 80)")
    p.add_argument("--water", type=float, default=25.0,
                   help="Water layer per side in Å (default: 25)")
    p.add_argument("--salt", type=float, default=0.15,
                   help="NaCl concentration in M (default: 0.15)")
    p.add_argument("--temperature", type=float, default=310.0,
                   help="Temperature in K (default: 310)")

    p.add_argument("--charmm-psf", metavar="PSF",
                   help="CHARMM-GUI PSF file (skips builder, loads all-atom system)")
    p.add_argument("--charmm-pdb", metavar="PDB",
                   help="CHARMM-GUI PDB file (required with --charmm-psf)")
    p.add_argument("--charmm-ff", nargs="*", default=[],
                   metavar="FILE",
                   help="Extra CHARMM FF files (.xml/.prm/.str)")

    p.add_argument("--output", default="membrane.pdb")
    p.add_argument("--workdir", default="membrane_build")
    p.add_argument("--no-equilibrate", action="store_true")

    p.add_argument("--stream", action="store_true",
                   help="Stream to PDB Visual via MDSS")
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--viewer-exe", default="pdbvisual")
    p.add_argument("--production-time", type=float, default=60.0,
                   help="Production wall-clock seconds (default: 60)")

    p.add_argument("--list-lipids", action="store_true",
                   help="Print supported lipid names and exit")

    args = p.parse_args()

    if args.list_lipids:
        print("\nSupported lipid residue names:")
        for n in sorted(_LIPID_NAMES_5):
            apl, _, _ = LIPID_INFO.get(n, _DEFAULT)
            print(f"  {n:6s}  area/lipid ≈ {apl:.0f} Å²")
        print(f"  {'CHOL':6s}  Cholesterol")
        print()
        return

    try:
        import openmm
        from openmm import LangevinMiddleIntegrator
        from openmm import MonteCarloBarostat
        from openmm.app import Simulation
        from openmm.unit import kelvin, bar, picoseconds, femtoseconds
    except ImportError:
        sys.exit("OpenMM not found:  conda install -c conda-forge openmm")

    os.makedirs(args.workdir, exist_ok=True)

    charmm_mode = bool(args.charmm_psf)

    print(f"\n{'='*55}")
    print(f"  Membrane Builder  —  {'CHARMM-GUI all-atom' if charmm_mode else 'CG 5-bead preview'}")
    if not charmm_mode:
        composition = parse_lipids(args.lipids)
        print(f"  Size     : {args.size[0]:.0f} × {args.size[1]:.0f} Å")
        print(f"  Water    : {args.water:.0f} Å / side")
        print(f"  Salt     : {args.salt} M NaCl")
    print(f"  Temp     : {args.temperature} K")
    print(f"{'='*55}\n")

    # ── Build or load ─────────────────────────────────────────────────────────
    if charmm_mode:
        if not args.charmm_pdb:
            sys.exit("[!] --charmm-pdb required with --charmm-psf")
        psf, positions = load_charmm_gui(args.charmm_psf, args.charmm_pdb)
        topology  = psf.topology
        system    = create_charmm_system(psf, args.charmm_ff)
        viewer_pdb = args.charmm_pdb

    else:
        # CG geometric builder
        topology, positions, box_nm = build_bilayer(
            composition, tuple(args.size), args.water
        )
        topology, positions = add_water_and_ions(
            topology, positions, box_nm, args.water, args.salt
        )

        # Save raw PDB for viewer
        from openmm.app import PDBFile
        viewer_pdb = os.path.join(args.workdir, "membrane_raw.pdb")
        with open(viewer_pdb, "w") as f:
            PDBFile.writeFile(topology, positions, f)
        print(f"[Build] Raw structure → {viewer_pdb}")

        ff     = get_cg_forcefield()
        system = create_cg_system(topology, ff, box_nm)

    # Barostat — only valid with PBC (CHARMM/PME mode); CG uses non-periodic
    if charmm_mode:
        try:
            from openmm import MonteCarloMembraneBarostat
            baro = MonteCarloMembraneBarostat(
                1.0 * bar, 0.0 * bar,
                args.temperature * kelvin,
                MonteCarloMembraneBarostat.XYIsotropic,
                MonteCarloMembraneBarostat.ZFree, 25,
            )
            print("[FF] Semi-isotropic NPT barostat")
        except Exception:
            baro = MonteCarloBarostat(1.0 * bar, args.temperature * kelvin, 25)
            print("[FF] Isotropic NPT barostat")
        system.addForce(baro)

    integrator = LangevinMiddleIntegrator(
        args.temperature * kelvin, 1.0 / picoseconds, 2.0 * femtoseconds
    )
    simulation = Simulation(topology, system, integrator)

    # ── MDSS reporter ─────────────────────────────────────────────────────────
    reporter = None
    if args.stream:
        sys.path.insert(0, str(Path(__file__).parent))
        from openmm_mdss import MDSSReporter
        exe = os.path.abspath(args.viewer_exe) if os.path.exists(args.viewer_exe) \
              else args.viewer_exe
        reporter = MDSSReporter(
            reportInterval = 50,
            port           = args.port,
            pdb_path       = os.path.abspath(viewer_pdb),
            launch_viewer  = True,
            viewer_exe     = exe,
        )

    # ── Equilibrate ───────────────────────────────────────────────────────────
    if not args.no_equilibrate:
        equilibrate(simulation, positions, args.temperature, reporter)
        save_pdb(simulation, topology, args.output)
        ckpt = args.output.replace(".pdb", ".chk")
        simulation.saveCheckpoint(ckpt)
        print(f"\n[Done] Equilibrated membrane:")
        print(f"       PDB        : {args.output}")
        print(f"       Checkpoint : {ckpt}")
        print(f"\n       Resume later:")
        print(f"       python scripts/openmm_mdss.py {args.output} "
              f"--live-stream {args.port}")
    else:
        save_pdb(simulation, topology, args.output)
        print(f"\n[Done] Raw membrane → {args.output}")

    # ── Production ────────────────────────────────────────────────────────────
    if args.stream and reporter and not args.no_equilibrate:
        simulation.reporters.append(reporter)
        print(f"\n[MD] Production ({args.production_time:.0f}s) — Ctrl+C to stop…")
        try:
            simulation.runForClockTime(args.production_time)
        except KeyboardInterrupt:
            print("\n[MD] Stopped.")
        except Exception as e:
            print(f"\n[MD] Error: {e}")
        reporter.close()

    if not charmm_mode:
        print("\nNote: CG preview uses 5 beads/lipid. For production simulations:")
        print("  1. Build with CHARMM-GUI: charmm-gui.org → Membrane Builder")
        print("  2. Download 'OpenMM' format, extract step5_input.psf + .pdb")
        print(f"  3. python scripts/create_membrane.py "
              f"--charmm-psf step5_input.psf --charmm-pdb step5_input.pdb "
              f"--stream --viewer-exe {args.viewer_exe}")


if __name__ == "__main__":
    main()
