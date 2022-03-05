'''Functions dealing with i/o operations'''

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
    atom_symbol = np.loadtxt(xyz_path, usecols=0, dtype='str', max_rows=atoms_num+2)

    if index == -1:
        _atoms = ''.join(molecules[index * atoms_num:])
    else:
        _atoms = ''.join(molecules[index * (atoms_num+2):index * (atoms_num+2) + atoms_num])

    return atoms_num, atom_symbol, _atoms

def write_xyz(xyz_path, atom_num, atoms):
    _atoms = np.reshape(atoms.split(), (atom_num,4))   

    with open(xyz_path,'a+') as xyz:
        last_line = xyz.readlines()[-1]
        if last_line[-1] != '\n':
            xyz.write('\n')
        else:
            pass
        xyz.write(str(atom_num)  )
        xyz.write('\n')
        for atom in _atoms:
            xyz.write('%8s %20.15f %20.15f %20.15f\n'%(atom[0], atom[1], atom[2], atom[3]))

class Data(object):
    def __init__(self, qc_engine, ml_engine, mainobject):
        
        if ml_engine.lower() == 'deepmd':
            if qc_engine is None or qc_engine.lower() == 'pyscf':
                data = PySCFdata
            elif qc_engine.lower() == 'gaussain':
                data = Gaussiandata
            elif qc_engine.lower() == 'vasp':
                data = VASPdata
            else:
                raise(NotImplementedError)
        else:
            raise(NotImplementedError)

        return data(mainobject)

class PySCFdata(object):# the input obect may need to be modified 1/22/2022
    '''generate input files for ML engine from PySCF results'''
    def __init__(self):
        self.mf       =    engine.QC_engine.mf
        self.mol      =    self.mf.mol
        self.atoms    =    self.mol.atom # a list includes atom symbol and coordinates
        self.coords   =    self.mol.atom_coords() # a np tuple
        self.energy   =    self.mf.energy_tot()
        self.forces   =    self.mf.Gradients().grad()
        
        self.ML_engine = mainobject.ML_engine
        self.file_path = mainobject.file_path

        # constants
        from pyscf.data.nist import BOHR ,HARTREE2EV
        bohr2ang   = BOHR
        hartree2eV = HARTREE2EV

        # data conversion 
        if self.ML_engine.lower() == 'deepmd':
            self.length_convert = bohr2ang
            self.energy_convert = hartree2eV
            self.force_convert  = hartree2eV / bohr2ang

    def dump(self):
        if self.ML_engine.lower() == 'deepmd':
            return self.dump_to_deepmd()

    def dump_to_deepmd(self): # dump data to raw file for MLe engine
        '''type_map.raw format:

        atom_symbol_1 atom_symbol_2 ...
    
        n            : integer, the index of atom
        atom_symbol_n: str, the symbol of atom
        '''

        with open(self.file_path + 'type_amp', 'w') as type:
            for i in range(0, len(self.coords)):
                type.write(self.mol.atom_symbol(i) + ' ')

        '''coord.raw format:
        
        coord_x_1 coord_y_1 coord_z_1 coord_x_2 coord_y_2 coord_z_2 ...

        coord_x_n, coord_y_n, coord_z_n: float, 3 components of coordinates of an atom
        '''
        with open(self.file_path + 'coord.raw', 'w') as coord:
            coord.write(str(self.coords.flatten() * self.length_convert))

        '''energy.raw format:
        
        energy_1

        energy_n: float, total energy of a system
        '''
        with open(self.file_path + 'energy.raw', 'w') as energy:
            energy.write(str(self.energy +  self.energy_convert))

        '''force.raw format:
        
        force_x_1 force_y_1 force_z_1 force_x_2 force_y_2 force_z_2 ...

        force_x_n, force_y_n, force_z_n: float, 3 components of force of an atom
        '''
        with open(self.file_path + 'force.raw', 'w') as force:
            force.write(str(self.forces.flatten() * self.force_convert))

        



        
