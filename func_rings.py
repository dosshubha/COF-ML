from ase.io import read, write

# import pandas as pd
import numpy as np
from ase.build import molecule
from ase.neighborlist import get_connectivity_matrix
from ase.neighborlist import natural_cutoffs
from ase.neighborlist import NeighborList
import glob
from tqdm import tqdm
from skspatial.objects import Plane, Points
from utils import *

######################################################
##Reding the the COFs
######################################################
sizes = []
formulas = []
tags = []
files = glob.glob("./63*cif")
tags_wmvo = []
tags_womvo = []
tags_wsp2n = []
tags_wosp2n = []
print(files)
for i in tqdm(files, total=len(files), desc="Progress"):
    atoms = read(i)
    # print(i)
    symbol = atoms.get_chemical_symbols()
    size = len(symbol)
    formula = atoms.get_chemical_formula()
    sizes.append(size)
    formulas.append(formula)
    tag = i[2:-4]  ##adjust this according to the source cifs
    # print(tag)
    tags.append(tag)
    ############################################################
    ### Block for functionaling the rings
    ###########################################################
    nl, cm = compute_ase_neighbour(atoms)
    graph = matrix2dict(cm)
    cutoffs = natural_cutoffs(atoms, mult=1.0)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    G = nx.from_numpy_array(cm)
    six_rings = find_rings(G, 6)
    five_rings = find_rings(G, 5)
    tot_rings = six_rings + five_rings
    #print(tot_rings)
    ##Adjusting for the Li already present in the cif
    Li_indices = []
    for idx, elements in enumerate(symbol):
        if elements == "Li":
            Li_indices.append(idx)
    rings_excl_Li = [rings for rings in tot_rings if not any(num in rings for num in Li_indices)]
    #print(rings_excl_Li)
    add_Li_to_rings(tag, atoms, rings_excl_Li, "./")
