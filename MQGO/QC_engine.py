'''Functions for QC energy and gradient calculation'''

import numpy as np
from pyscf import gto, scf, dft, grad
from pyscf.data.nist import BOHR
from MQGO.data import read_xyz

class QCEngine(): 
    def __init__(self, qc_engine=None, xyz_path=None, **setting):

        assert qc_engine is not None, 'Please specify which QC engine to use.'
        assert xyz_path is not None, 'Please specify the path of xyz file.'
        assert setting is not None, 'Please specify settings for QC calcualations.'

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
    

class PySCF(): # moleclue is the Mole object of gto module
    '''Create PySCF mol and mf object and run energy and gradient calculation
    Usage:
        qcengine = PySCF(xyz_path='xxx', a=xxx, b=xxx,...)
        energy, force = qcengine.calc_new()   
    '''
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

        self.e_tot  = None
        self.force  = None
        self.coords = self.mol.atom_coords()
        self.dm0    = None

    def build_mf_object(self):
        '''Build mf object for DFT or HF calculation''' # TDSCF calculation support can be done in the future 3/8/2022
        # check setting
        scf_basic_keys = ['xc', 'restricted'] # key restricted is bool
        scf_advance_keys = ['conv_tol', 'max_cycle', 'verbose', 'grids.level', 'dm0']
        scf_default_dict = {'conv_tol':1e-12, 'max_cycle':100, 'verbose':0, 'grids.level':3, 'dm0':None}
        for key in scf_basic_keys:
            assert key in self.keys, "Keyword '%s' mmust be specified, please check setting."%(key)

        for key in scf_advance_keys:
            if key not in self.keys:
                self.setting[key] = scf_default_dict[key]

        # build mf object
        if self.setting['xc'] is not None:
            if self.setting['restricted'] == True or self.setting['restricted'] == 1:
                self.mf = dft.RKS(self.mol)
                self.mf.xc = self.setting['xc']
            elif self.setting['restricted'] == False or self.setting['restricted'] == 0:
                self.mf = dft.UKS(self.mol)
                self.mf.xc = self.setting['xc']
        
        elif self.setting['xc'] is None:
            if self.setting['restricted'] == True or self.setting['restricted'] == 1:
                self.mf = scf.RHF(self.mol)
            elif self.setting['restricted'] == False or self.setting['restricted'] == 0:
                self.mf = scf.UHF(self.mol)

        

        self.mf.conv_tol    = self.setting['conv_tol']
        self.mf.max_cycle   = self.setting['max_cycle']
        self.mf.verbose     = self.setting['verbose']
        self.mf.grids.level = self.setting['grids.level']            

    def check_scf_converge(self):
        assert self.mf.converged is True, 'SCF is not converged, please modify related paramaters and rerun the calculations.'

    def calc_new(self):
        # run calculation
        self.build_mf_object()
        # print('QC_engine.py 108:',self.mf.mol.atom_coords())
        self.mf.kernel(dm0=self.dm0)
        if self.setting['dm0'] is not None:
            self.dm0 = self.mf.make_rdm1()
            
        self.check_scf_converge()

        self.e_tot = self.mf.e_tot
        self.force = self.mf.Gradients().grad()

        return self.e_tot, self.force

    def update_coord(self, new_coord):
        self.mol = self.mol.set_geom_(new_coord)
        self.coords = self.mol.atom_coords()
        # print('QC_engine.py 118:', self.mol.atom_coords()*BOHR)
        # self.mf.reset(self.mol)

class Gaussian(object):
    
    def __init__(self, xyz_path=None, **setting):
        return NotImplemented

    def write_input_file(self):
        return NotImplemented

    def read_output_file(self):
        return NotImplemented

class VASP(object):
    
    def __init__(self, xyz_path=None, **setting):
        return NotImplemented

    def write_input_file(self):
        return NotImplemented

    def read_output_file(self):
        return NotImplemented