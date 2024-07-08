#/usr/bin/python
from ase.io import read, write
import numpy as np
from ase.build import molecule
from ase.neighborlist import get_connectivity_matrix
from ase.neighborlist import natural_cutoffs
from ase.neighborlist import NeighborList
import glob
from mofstructure import mofdeconstructor
import networkx as nx
from skspatial.objects import Plane, Points

filenames=glob.glob('./*cif')
filenames=filenames[2:]

atoms = read("./31.cif")
atoms_120 = read("./120.cif")
atoms_256 = read("./256.cif")
atoms_993 = read("./993.cif")
atoms_1 = read("./1.cif")
#print(atoms.get_chemical_symbols())
nl, cm  = mofdeconstructor.compute_ase_neighbour(atoms)
#print(nl)
#print(nl[102])
np.save('cm.npy',cm)
graph = mofdeconstructor.matrix2dict(cm)
#print(graph)

cutoffs = natural_cutoffs(atoms_256, mult=1.0)
nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
nl.update(atoms_256)
#print(nl)


def unit_vector(v):
    return v / np.linalg.norm(v)

def angle_between(u, v):
    v_1 = unit_vector(v)
    u_1 = unit_vector(u)
    return np.arccos(np.clip(np.dot(v_1, u_1), -1.0, 1.0))

def find_rings(graph, ringsize):
    ri = nx.simple_cycles(graph, length_bound=ringsize)
    rings = []
    for ring in ri:
        if len(ring) ==ringsize:
           rings.append(ring)
    return rings

def compute_CN(atoms,idx):
    symbols =  atoms.get_chemical_symbols()
    cutoffs = natural_cutoffs(atoms, mult=0.9)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    coenv_indices, offsets = nl.get_neighbors(idx)
    CN = len(coenv_indices)
    coenv = []
    for s in coenv_indices:
        coenv.append(symbols[s])
    return CN, coenv_indices, coenv
    
    
    


def find_NO(atoms):
    symbols =  atoms.get_chemical_symbols()
    o_indices = []
    n_indices = []
    for idx, elements in enumerate(symbols):
        if elements == "O":
            o_indices.append(idx)
        if elements == "N":
            n_indices.append(idx)
    sp2_o = []
    sp3_o = []
    sp2_n = []
    sp3_n = []
    sp_n = []
    for i in o_indices:
        CN, coenv_indices,coenv = compute_CN(atoms,i)
        if CN == 1:
            sp2_o.append(i)
        elif CN == 2:
            for j in range(len(coenv)):
                if coenv[j]== "H":
                    CN_H, coenv_indices_H,coenv_H = compute_CN(atoms,coenv_indices[j])
                    if CN_H == 2:
                        sp2_o.append(i)			
        else:
            sp3_o.append(i)
    for i in n_indices:
        CN, coenv_indices,coenv = compute_CN(atoms,i)
        CN = len(coenv_indices)
        if CN == 2:
            sp2_n.append(i)
        elif CN ==3:
            sp3_n.append(i)
        else:
            sp_n.append(i)
    return sp2_o, sp2_n

#sp2_o, = find_NO(atoms_256)
sp2_o, sp2_n = find_NO(atoms_1)
print("The indices fod the sp2 oxygens are:", sp2_o)
print("The indices fod the sp2 nitrogen are:", sp2_n)

def add_Li_to_mvo(name,atoms,indices):
    Lipos = []
    for i in indices:
        subs=[]
        vecs=[]
        pos = atoms.get_positions()[i]
        pos = np.around(pos,4)
        pos[2] = pos[2]+ 1.8
        Lipos.append(pos)
    for coord in Lipos:
        atoms.append('Li')
        atoms.positions[-1] = coord
    write(f"{name}_mvoLi.cif",atoms,format='cif')

def add_Li_to_sp2n(name,atoms,indices):
    Lipos = []
    for i in indices:
        CN, coenv_indices,coenv = compute_CN(atoms,i)
        subs=[]
        vecs=[]
        sp2_npos = atoms.get_positions()[i]
        print("the sp2 N position:", sp2_npos)
        for j in coenv_indices:
            pos = atoms.get_positions()[j]
            pos = np.around(pos,4)
            subs.append(pos)
        subs.append(sp2_npos)
        points = Points(subs)
        #try
        #plane = Plane.best_fit(points)
        #vec = plane.normal
        vec = [0.0, 0.0, 0.0]
        midpoint = np.mean(subs, axis=0)
        vec_along_n = unit_vector(sp2_npos - midpoint)
        print("the normal vector is:", vec)
        d = 1.428
        vec2 = sp2_npos + d * vec_along_n
        vec[0] = vec2[0]
        vec[1] = vec2[1]
        vec[2] =sp2_npos[2] + 1.4
        Lipos.append(vec)
    for coord in Lipos:
        atoms.append('Li')
        atoms.positions[-1] = coord
    write(f"{name}_sp2nLi.cif",atoms,format='cif')


#add_Li_to_mvo('256',atoms_256,sp2_o)
add_Li_to_sp2n('1',atoms_1,sp2_n)
#for coord in vectors:
#    #print(coord)
#    atoms_120.append('Li')
#    atoms_120.positions[-1] = coord
#print("The normal vectors are", vectors)
#print(atoms)
#write(f"{name}_mvoLi.cif",atoms,format='cif')


G = nx.from_numpy_array(cm)
six_rings = find_rings(G, 6)
#print(six_rings)

def add_Li(ase_atoms, rings):
    Lipos = []
    for ring in rings:
        #print(ring)
        subs = []
        vecs = []
        for at in ring[0:5:2]:
            #print(at)
            pos = ase_atoms.get_positions()[at]
            #print(pos)
            pos = np.around(pos, 4)
            subs.append(pos)
            #print(subs)
        #print(subs)
        points = Points(subs)
        plane = Plane.best_fit(points)
        vec = plane.normal
        vec = vec * 2.0
        midpoint = np.mean(subs, axis = 0)
        #print("MIDPOINT",midpoint)
        vec = midpoint + vec
        #print(vec)
        #vecs.append(vec)
        Lipos.append(vec)
    return Lipos

vectors = add_Li(atoms,six_rings)
for coord in vectors:
    #print(coord)
    atoms.append('Li')
    atoms.positions[-1] = coord
#print("The normal vectors are", vectors)
#print(atoms)
write('./31-Li.cif',atoms,format='cif')
		

		
        
