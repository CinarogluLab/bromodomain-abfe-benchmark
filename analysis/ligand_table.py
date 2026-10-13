"""SI Table S4 rows from ligands.tsv.
"""
import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

from pairwise_ddg import BRD, PDB

HERE = Path(__file__).resolve().parent
LATEX_SPECIAL = {"#": r"\#", "%": r"\%", "&": r"\&", "_": r"\_", "$": r"\$", "{": r"\{", "}": r"\}"}


def breakable_smiles(smi):
    """Monospace SMILES that may wrap after any character."""
    return r"\texttt{" + r"\allowbreak{}".join(LATEX_SPECIAL.get(c, c) for c in smi) + "}"


def main():
    with open(HERE / "ligands.tsv") as fh:
        smiles = {r["pdb"]: r["smiles_as_used_for_fingerprints"].strip() for r in csv.DictReader(fh, delimiter="\t")}
    rows = []
    for pdb, brd in zip(PDB, BRD):
        mol = Chem.MolFromSmiles(smiles[pdb])
        rows.append(f"{pdb} & {brd} & {breakable_smiles(smiles[pdb])} & {mol.GetNumHeavyAtoms()} & "
                    f"{Descriptors.MolWt(mol):.1f} \\\\")
    (HERE.parent / "tableS4_rows.tex").write_text("\n".join(rows) + "\n")
    print("wrote tableS4_rows.tex")


if __name__ == "__main__":
    main()
