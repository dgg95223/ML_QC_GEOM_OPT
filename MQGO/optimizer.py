'''Geometry optimization module of ASE'''

import subprocess
from ase.io import read, write
from MQGO.data import xyz_write_check, read_xyz

class Optimizer():
    def __init__(self, xyz_path=None, work_path=None, ml_engine=None, outter_cycle=None, algorithm=None, max_opt_cycle=None):
        assert work_path is not None, 'Please specify the path for storing output files.'
        assert xyz_path is not None, 'Please specify the path of xyz file.'
        assert ml_engine is not None, 'Please specify the engine of calculator.'

        self.xyz_path = xyz_path
        self.ml_engine = ml_engine
        self.work_path = work_path

        # self.atom_symbol = atom_symbol
        self.outter_cycle = outter_cycle
        algorithms = ['bfgs', 'lbfgs', 'gpmin', 'pyberny']  # pyberny need to be installed separatedly

        self.pes_file_path = work_path
        self.xyz_path = xyz_path

        if algorithm is None:
            self.geom_opt_algorithm = 'bfgs'
        else:
            assert algorithm in algorithms, 'The selected algorithm is not supported with ASE optimizer.'
            self.geom_opt_algorithm = algorithm

        if max_opt_cycle is None:
            self.max_opt_cycle = 100
        else:
            self.max_opt_cycle = max_opt_cycle

        self.geom_opt    = None
        self.ene_opt     = None
        self.force_opt   = None
        self.conv_tol    = None
        self.global_temp = None
        self.target_geom = 0 # optimize the first geometry as defaut

    def ase_read_any_xyz(self, index=None):
        if index is None:
            index = -1

        atom_num, atom_symbol, atom_coord = read_xyz(self.xyz_path, index=index, output='pyscf')
        with open('./_temp.xyz', 'r') as temp:
            temp.write(str(atom_num)+'\n')
            temp.write('\n')
            temp.write(atom_coord)
        
        _atoms = self.ase_read_xyz('./_temp.xyz')
        subprocess.run('rm ./_temp.xyz', shell=True)
        return _atoms

    def ase_read_xyz(self, xyz_file=None): # Build 'Atoms'object from xyz file, a xyz file may contain multiple geometries
        ''' ASE's Atoms object usage: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
        For current implementation, each geometry is read/written in a single xyz file, an aternative way
        is to write all geometries in one file.
        '''
        if xyz_file is None:
            xyz_file = self.xyz_file
        else:
            pass
        atoms_obj = read(xyz_file, index=self.target_geom)
        return atoms_obj

    def ase_write_xyz(self, atom_obj):
        xyz_write_check(self.xyz_path)
        write(self.xyz_path, atom_obj, append=True)

    def run_opt(self, ml_engine):
        '''Reference: https://wiki.fysik.dtu.dk/ase/ase/optimize.html
        '''
        opt_global = False
        if self.geom_opt_algorithm == 'bfgs':
            from ase.optimize import BFGS
            optimizer_ = BFGS
        elif self.geom_opt_algorithm == 'lbfgs':
            from ase.optimize import LBFGS
            optimizer_ = LBFGS
        elif self.geom_opt_algorithm == 'gpmin':
            from ase.optimize import GPMin
            optimizer_ = GPMin
        elif self.geom_opt_algorithm == 'pyberny':
            from ase.optimize import Berny
            optimizer_ = Berny
        elif self.geom_opt_algorithm == 'basin':
            from ase.optimize.basin import BasinHopping
            from ase.optimize import LBFGS
            optimizer_ = BasinHopping
        else:
            raise(NotImplementedError)

        if self.geom_opt_algorithm == 'basin':
            opt_global = True
            if self.global_temp is None:
                self.global_temp = 100

        _atoms = self.ase_read_any_xyz()   # choose any geometry to optimize , the default is the last geometry in the xyz file

        print('optimizer.py 77:', _atoms.get_positions())
        if ml_engine.lower() == 'deepmd':
            from deepmd.calculator import DP
            _atoms.calc = DP(model=self.pes_file_path+'graph-compress.pb')

        if opt_global is not True:
            # local optimization
            opt = optimizer_(_atoms) 
            opt.run(fmax=self.conv_tol)
            
        else:
            # global optimization
            from ase.units import kB                 # need further test 3/15/2022
            opt = optimizer_(atoms=_atoms,           # the system to optimize
                  temperature=self.global_temp * kB, # 'temperature' to overcome barriers
                  dr=0.5,                            # maximal stepwidth
                  optimizer=LBFGS,                   # optimizer to find local minima
                  fmax=self.conv_tol,                # maximal force for the optimizer
                  )

        self.ase_write_xyz(_atoms)
        self.geom_opt  = _atoms.get_positions()
        self.ene_opt   = _atoms.get_potential_energy()
        self.force_opt = _atoms.get_forces()