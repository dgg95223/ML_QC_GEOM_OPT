'''Functions dealing with i/o operations'''
from ast import Try
import subprocess, os
import numpy as np

def read_xyz(xyz_path, index=None):
    '''Read geometry from a xyz file with multiple geometries'''
    # One geometry in one xyz file
    # col1 = np.loadtxt(xyz_path, usecols=0, dtype='str')
    # atoms_num = int(col1[0])
    # atom_symbol = col1[1:]
    # with open(xyz_path, 'r') as xyz:
    #     _atoms_  = xyz.readlines()
    # _atoms = ''.join(_atoms_[2:]) # A string containing atom symbol and coordinates
    
    # return atoms_num, atom_symbol, _atoms
    
    ####################################################################################

    # Multiple geometries in one xyz file
    if index is None:
        index = -1  # read the last geometry as default
    elif index > 0:
        index = index - 1
    elif index == 0:
        assert index != 0, "'index' should be a natural number"

    with open(xyz_path,'r') as xyz:
        molecules = xyz.readlines()
    
    atoms_num = int(molecules[0])
    atom_symbol = np.loadtxt(xyz_path, usecols=0, dtype='str', max_rows=atoms_num+2)[1:]

    if index == -1:
        _atoms = ''.join(molecules[index * atoms_num:])
    elif molecules[(index + 1) * (atoms_num + 2) - 1][-1] == '\n':
        _atoms = ''.join(molecules[index * (atoms_num + 2) + 2 :(index + 1) * (atoms_num + 2)])[:-1]
    else:
        _atoms = ''.join(molecules[index * (atoms_num + 2) + 2 :(index + 1) * (atoms_num + 2)])
    return atoms_num, atom_symbol, _atoms

def write_xyz(xyz_path, atom_num, atoms):
    _atoms = np.reshape(atoms.split(), (atom_num,4))   

    with open(xyz_path,'r+') as xyz:
        last_line = xyz.readlines()[-1]
        if last_line[-1] != '\n':
            xyz.write('\n')
        else:
            pass
        xyz.write(str(atom_num))
        xyz.write('\n')
        for atom in _atoms:
            xyz.write('%2s %20.15f %20.15f %20.15f\n'%(atom[0], 
                                                    np.float64(atom[1]),
                                                    np.float64(atom[2]),
                                                    np.float64(atom[3])))

def dump_deepmd_raw(work_path, atom_symbol, coords, energy, forces, pbc=False, append=True):
    raw_path = work_path+'raw/'
    have_raw_path = os.path.exists(raw_path)
    if have_raw_path:
        pass
    else:
        os.makedirs(raw_path)

    if append is False:
        mode = 'w+'
    else:
        mode = 'a+'

    _atom_symbol = list(set(atom_symbol))  # a list of atom symbol without repeating element
    type_dict = {}
    for ii, i in enumerate(_atom_symbol):
        type_dict[i] = ii

    '''type.raw format:

    atom_index_1 atom_index_2 ...

    n            : integer, the index of atom
    atom_index_n: str, the index of atom
    '''

    with open(raw_path + 'type.raw', mode) as type:
        if mode == 'a+':
            type.write('\n')
        else:
            pass
        for i in atom_symbol:
            type.write(str(type_dict[i]) + '')

    '''type_map.raw format:

    atom_symbol_1 atom_symbol_2 ...

    n            : integer, the index of atom
    atom_symbol_n: str, the symbol of atom
    '''

    with open(raw_path + 'type_amp.raw', mode) as type:
        if mode == 'a+':
            type.write('\n')
        else:
            pass
        for i in range(0, len(_atom_symbol)):
            type.write(_atom_symbol[i] + '')

    '''coord.raw format:
    
    coord_x_1 coord_y_1 coord_z_1 coord_x_2 coord_y_2 coord_z_2 ...

    coord_x_n, coord_y_n, coord_z_n: float, 3 components of coordinates of an atom
    Each line contains coordinates of all atoms in one frame
    '''
    with open(raw_path + 'coord.raw', mode) as coord:
        if mode == 'a+':
            coord.write('\n')
        else:
            pass
        for i in coords.flatten():
            coord.write('%15.11f'%i + ' ')

    '''energy.raw format:
    
    energy_1

    energy_n: float, total energy of a system
    '''
    with open(raw_path + 'energy.raw', mode) as ene:
        if mode == 'a+':
            ene.write('\n')
        else:
            pass
        ene.write(str(energy))

    '''force.raw format:
    
    force_x_1 force_y_1 force_z_1 force_x_2 force_y_2 force_z_2 ...

    force_x_n, force_y_n, force_z_n: float, 3 components of force of an atom
    '''
    with open(raw_path + 'force.raw', mode) as force:
        if mode == 'a+':
            force.write('\n')
        else:
            pass
        for i in forces.flatten():
            force.write('%20.15f'%i + ' ')

    if pbc is False:
        with open(raw_path+'nopbc', 'w') as pbc:
            pass
    else: # PBC is not implemented
        pass

def dump_deepmd_npy(work_path, pbc=False):
    if pbc is True:
        raise NotImplemented

    raw_path = work_path+'raw/'
    npy_path = work_path+'raw/set.000/'  # need to be improved in the future
    have_npy_path = os.path.exists(npy_path)
    if have_npy_path:
        pass
    else:
        os.makedirs(npy_path)
    
    coord = np.loadtxt(raw_path+'coord.raw')
    energy = np.loadtxt(raw_path+'energy.raw')
    forces = np.loadtxt(raw_path+'forces.raw')

    np.save(npy_path+'coord.npy', coord)
    np.save(npy_path+'energy.npy', energy)
    np.save(npy_path+'forces.npy', forces)

class Data():
    '''This class control the data communication between QC engine and ML engine'''
    def __init__(self, qc_engine, ml_engine, work_path):
        
        self.qc_engine = qc_engine
        self.ml_engine = ml_engine
        self.work_path = work_path
        
        if ml_engine.lower() == 'deepmd':
            if qc_engine is None or qc_engine.lower() == 'pyscf':
                self.data = PySCFdata
            elif qc_engine.lower() == 'gaussain':
                self.data = Gaussiandata
            elif qc_engine.lower() == 'vasp':
                self.data = VASPdata
            else:
                raise(NotImplementedError)
        else:
            raise(NotImplementedError)

    def build(self, engine_obj=None):
        assert engine_obj is not None, 'Please specify a engine object which is the data source.'
        return self.data(self.ml_engine, self.work_path, engine_obj)


class PySCFdata():# the input obect may need to be modified 1/22/2022
    '''This class control the data communication between PySCF and ML engine'''
    def __init__(self, ml_engine, work_path, PySCF):
        self.mf          =    PySCF.mf
        self.mol         =    self.mf.mol
        self.atoms       =    self.mol.atom # a list includes atom symbol and coordinates
        self.atom_symbol =    PySCF.atom_symbol
        self.coords      =    self.mol.atom_coords() # a np tuple
        self.energy      =    self.mf.energy_tot()
        self.forces      =    self.mf.Gradients().grad()
        
        self.ML_engine = ml_engine
        self.work_path = work_path

        # constants
        from pyscf.data.nist import BOHR ,HARTREE2EV
        bohr2ang   = BOHR
        hartree2eV = HARTREE2EV

        # data conversion 
        if self.ML_engine.lower() == 'deepmd':
            self.length_convert = bohr2ang
            self.energy_convert = hartree2eV
            self.force_convert  = hartree2eV / bohr2ang

    def dump(self, append=True): # dump pyscf data to raw file for ML engine
        if self.ML_engine.lower() == 'deepmd':
            self.dump_to_deepmd(append=append)

    def dump_to_deepmd(self, append=True): # dump pyscf data to raw file for deepmd-kit  

        dump_deepmd_raw(self.work_path,
                         self.atom_symbol,
                         self.coords * self.length_convert,
                         self.energy * self.energy_convert,
                         self.forces * self.force_convert, 
                         append=append)
        dump_deepmd_npy(self.work_path, pbc=False)
        
class Gaussiandata():
    def __init__(self):
        pass
class VASPdata():
    def __init__(self):
        pass
        
        
