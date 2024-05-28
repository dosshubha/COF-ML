from ase.io import read, write
import pandas as pd
import numpy as np
from ase.build import molecule
from ase.neighborlist import get_connectivity_matrix
from ase.neighborlist import natural_cutoffs
from ase.neighborlist import NeighborList
import glob

sizes = []
formulas = []
tags = []
files = glob.glob("./CoRE-COFs_1242-v7.0/*cif")
for i in files:
    atoms = read(i)
    symbol = atoms.get_chemical_symbols()
    size = len(symbol)
    formula = atoms.get_chemical_formula()
    sizes.append(size)
    formulas.append(formula)
    tags.append(i[22:-4])
    print("Structure" + i[22:-4] + "with chemical formula"+ formula + "is processed")
df13 = pd.DataFrame(data=[tags,formulas,sizes]).T
df13.to_csv("./sizes.csv", index=None, header=None)
