from ase.io import read, write
import numpy as np
from ase.build import molecule
from ase import neighborlist, geometry
from ase.neighborlist import get_connectivity_matrix
from ase.neighborlist import natural_cutoffs
from ase.neighborlist import NeighborList
import glob
import networkx as nx
from skspatial.objects import Plane, Points


def unit_vector(v):
    """Determines unit vector"""
    return v / np.linalg.norm(v)


def angle_between(u, v):
    """Calculate the angle between two vectors using the dot product formula"""
    v_1 = unit_vector(v)
    u_1 = unit_vector(u)
    return np.arccos(np.clip(np.dot(v_1, u_1), -1.0, 1.0))


def compute_CN(atoms, idx):
    """Computes the corrdination number and local environment of an atom from the ASE atom object and
    the index of the atom as the args"""
    symbols = atoms.get_chemical_symbols()
    cutoffs = natural_cutoffs(atoms, mult=0.9)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    coenv_indices, offsets = nl.get_neighbors(idx)
    CN = len(coenv_indices)
    coenv = []
    for s in coenv_indices:
        coenv.append(symbols[s])
    return CN, coenv_indices, coenv


def compute_ase_neighbour(ase_atom):
    """
    Borrowed from mofstructure
    Create a connectivity graph using ASE neigbour list.

    Parameters:
    -----------
    ASE atoms

    Returns
    -------
    1.  atom_neighbors:
        A python dictionary, wherein each atom index
        is key and the value are the indices of it neigbours.
        e.g.
        atom_neighbors ={0:[1,2,3,4], 1:[3,4,5]...}

    2.  matrix:
        An adjacency matrix that wherein each row correspond to
        to an atom index and the colums correspond to the interaction
        between that atom to the other atoms. The entries in the
        matrix are 1 or 0. 1 implies bonded and 0 implies not bonded.

    """
    atom_neighbors = {}
    cut_off = neighborlist.natural_cutoffs(ase_atom)

    neighbor_list = neighborlist.NeighborList(
        cut_off, self_interaction=False, bothways=True
    )
    neighbor_list.update(ase_atom)
    matrix = neighbor_list.get_connectivity_matrix(sparse=False)

    for atoms in ase_atom:
        connectivity, _ = neighbor_list.get_neighbors(atoms.index)
        atom_neighbors[atoms.index] = connectivity

    return atom_neighbors, matrix


def matrix2dict(bond_matrix):
    """
    Borrowed from mofstructure
    A simple procedure to convert an adjacency matrix into
    a python dictionary.

    Parameters:
    -----------
    bond matrix : adjacency matrix, type: nxn ndarray

    Returns
    -------
    graph: python dictionary
    """
    graph = {}
    for idx, row in enumerate(bond_matrix):
        temp = []
        for r in range(len(row)):
            if row[r] != 0:
                temp.append(r)
        graph[idx] = temp
    return graph


def find_rings(graph, ringsize):
    """Finds ring in a structure given the graph and the ringsize as the args.
    Returns the indices of the rings as arrays"""
    ri = nx.simple_cycles(graph, length_bound=ringsize)
    rings = []
    for ring in ri:
        if len(ring) == ringsize:
            rings.append(ring)
    return rings


def find_NO(atoms):
    """Finds the sp2 N and O atom indices from the ase atom object as the arg"""
    symbols = atoms.get_chemical_symbols()
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
        CN, coenv_indices, coenv = compute_CN(atoms, i)
        if CN == 1:
            sp2_o.append(i)
        elif CN == 2:
            for j in range(len(coenv)):
                if coenv[j] == "H":
                    CN_H, coenv_indices_H, coenv_H = compute_CN(atoms, coenv_indices[j])
                    if CN_H == 2:
                        sp2_o.append(i)
        else:
            sp3_o.append(i)
    for i in n_indices:
        CN, coenv_indices, coenv = compute_CN(atoms, i)
        CN = len(coenv_indices)
        if CN == 2:
            sp2_n.append(i)
        elif CN == 3:
            sp3_n.append(i)
        else:
            sp_n.append(i)
    return sp2_o, sp2_n


def add_Li_to_mvo(name, atoms, indices, dest):
    """Functionalize with Li atoms from the name of the cif file, corresponding
    ase atom object and the sp2 o atom indices
    Args
    name: string, name of the cif file
    atoms: object, ASE ato object
    indices: array, indices of sp2 o in the cif unit cell
    dest: string, desitantion path for the functionalized cofs"""
    Lipos = []
    for i in indices:
        subs = []
        vecs = []
        pos = atoms.get_positions()[i]
        pos = np.around(pos, 4)
        pos[2] = pos[2] + 1.8
        Lipos.append(pos)
    for coord in Lipos:
        atoms.append("Li")
        atoms.positions[-1] = coord
    write(f"{dest}{name}_mvoLi.cif", atoms, format="cif")
    write(f"{dest}{name}_mvoLi.vasp", atoms, format="vasp")


def add_Li_to_sp2n(name, atoms, indices, dest):
    """Functionalize with Li atoms from the name of the cif file, corresponding
    ase atom object and the sp2 N atom indices
    Args
    name: string, name of the cif file
    atoms: object, ASE ato object
    indices: array, indices of sp2 N in the cif unit cell
    dest: string, desitantion path for the functionalized cofs"""
    Lipos = []
    for i in indices:
        CN, coenv_indices, coenv = compute_CN(atoms, i)
        subs = []
        vecs = []
        sp2_npos = atoms.get_positions()[i]
        # print("the sp2 N position:", sp2_npos)
        for j in coenv_indices:
            pos = atoms.get_positions()[j]
            pos = np.around(pos, 4)
            subs.append(pos)
        subs.append(sp2_npos)
        try:
            points = Points(subs)
            plane = Plane.best_fit(points)
            vec = plane.normal
        except Exception as m:
            vec = [0.0, 0.0, 0.0]
        midpoint = np.mean(subs, axis=0)
        vec_along_n = unit_vector(sp2_npos - midpoint)
        # print("the normal vector is:", vec)
        d = 1.428
        vec2 = sp2_npos + d * vec_along_n
        vec[0] = vec2[0]
        vec[1] = vec2[1]
        vec[2] = sp2_npos[2] + 1.4
        Lipos.append(vec)
    for coord in Lipos:
        atoms.append("Li")
        atoms.positions[-1] = coord
    write(f"{dest}{name}_sp2nLi.cif", atoms, format="cif")
    write(f"{dest}{name}_sp2nLi.vasp", atoms, format="vasp")


def add_Li_to_rings(name, atoms, rings, dest):
    """Functionalize all five and six membered rings in the cif with a Li"""
    Lipos = []
    for ring in rings:
        # print(ring)
        subs = []
        vecs = []
        for at in ring[0:3:1]:
            # print(at)
            pos = atoms.get_positions()[at]
            # print(pos)
            pos = np.around(pos, 4)
            subs.append(pos)
            # print(subs)
        # print(subs)
        points = Points(subs)
        plane = Plane.best_fit(points)
        vec = plane.normal
        vec = vec * 2.0
        midpoint = np.mean(subs, axis=0)
        # print("MIDPOINT",midpoint)
        vec = midpoint + vec
        # print(vec)
        # vecs.append(vec)
        Lipos.append(vec)
    for coord in Lipos:
        atoms.append("Li")
        atoms.positions[-1] = coord
    write(f"{dest}{name}_ringsLi.cif", atoms, format="cif")
