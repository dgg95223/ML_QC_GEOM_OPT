'''Functions for QC energy and gradient calculation'''

import numpy as np
from pyscf import gto, scf, dft, grad
from pyscf.data.nist import BOHR
from io import read_xyz

class QCEngine(object): 
    def __init__(self, qc_engine=None, xyz_path=None, **setting):
        self.qc_engine = qc_engine
        self.xyz_path  = xyz_path
        self.setting   = setting

    def build(self):
        if self.qc_engine is None or self.qc_engine.lower() == 'pyscf':
            engine = PySCF
        elif self.qc_engine.lower() == 'gaussain':
            engine = Gaussian
        elif self.qc_engine.lower() == 'vasp':
            engine = VASP

        return engine(xyz_path=self.xyz_path, **self.setting)

class PySCF(object): # moleclue is the Mole object of gto module
    '''create PySCF mol object and run energy and gradient calculation'''
    def __init__(self, xyz_path=None, **setting):
        # from pyscf import gto
        
        assert xyz_path is not None, "Can not find the xyz file"
        self.keys = []
        self.setting = setting

        # check mol keys  (custom basis set is not implemented in this version 1/24/2022)
        mol_basic_keys = ['basis', 'spin']
        # mol_default_keys = ['atoms', 'basis']
        mol_advance_keys = ['ecp', 'symmetry']

        for key in setting:
            self.keys.append(key.lower()) 

        for key in mol_basic_keys:
            assert key in self.keys, "Keyword '%s' mmust be specified, please check setting."%(key)

        for key in mol_advance_keys:
            if key not in self.keys:
                self.setting[key] = None

        # read xyz file
        self.atoms_num, self.atom_symbol, self.atoms = read_xyz(xyz_path)
        
        # build mol object
        self.mol = gto.Mole(atom=self.atoms, 
                    basis=self.setting['basis'], # to be improved for custome basis set
                    ecp=self.setting['ecp'], 
                    symmetry=self.setting['symmetry'],
                    spin=self.setting['spin']).build()

    def build_mf_object(self):
        # check setting
        scf_basic_keys = ['xc', 'restricted'] # key restricted is bool
        scf_advance_keys = ['conv_tol', 'max_cycle', 'verbose', 'grids.level']
        scf_default_dict = {'conv_tol':1e-12, 'max_cyccle':100, 'verbose':0, 'grids.level':3}
        for key in scf_basic_keys:
            assert key in self.keys, "Keyword '%s' mmust be specified, please check setting."%(key)

        for key in scf_advance_keys:
            if key not in self.keys:
                self.setting[key] = scf_default_dict[key]

        # build mf object
        if self.setting['xc'] is not None:
            if self.setting['restricted'] == True or self.setting['restricted'] == 1:
                self.mf = dft.RKS(self.mol)
            elif self.setting['restricted'] == False or self.setting['restricted'] == 0:
                self.mf = dft.UKS(self.mol)
        
        elif self.setting['xc'] is None:
            if self.setting['restricted'] == True or self.setting['restricted'] == 1:
                self.mf = scf.RHF(self.mol)
            elif self.setting['restricted'] == False or self.setting['restricted'] == 0:
                self.mf = scf.UHF(self.mol)

        self.mf.conv_tol   = self.setting['conv_tol ']
        self.mf.max_cycle  = self.setting['max_cycle']
        self.mf.verbose    = self.setting['verbose']
        # self.mf.grids.level = self.setting['grids.level']            

    def check_scf_converge(self):
        assert self.mf.converged is True, 'SCF is not converged, please modify related paramaters and rerun the calculations.'

    def cal_new(self):
        # run calculation
        self.mf.kernel()

        e_tot = self.mf.e_tot
        force = self.mf.Gradients().grad()

        return e_tot, force

    def update_coord(self, new_coord):
        self.mol = self.mol.set_geom_(new_coord * BOHR, inplace=True)
        self.mf.reset(self.mol)

class Gaussian(object):
    
    def __init__(self, xyz_path=None, **setting):

    def write_input_file(self):

    def read_output_file(self):
        return

class VASP(object):
    
    def __init__(self, xyz_path=None, **setting):

    def write_input_file(self):

    def read_output_file(self):
        return