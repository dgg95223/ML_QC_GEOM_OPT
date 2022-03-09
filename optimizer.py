'''Geometry optimization module of ASE'''

from ase.io import read, write

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

        self.geom_opt = None


    def ase_read_xyz(self): # Build 'Atoms'object from xyz file, a xyz file may contain multiple geometries
        ''' ASE's Atoms object usage: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
        For current implementation, each geometry is read/written in a single xyz file, an aternative way
        is to write all geometries in one file.
        '''
        
        atoms_obj = read(self.xyz_path, index=self.cycle + 1)
        return atoms_obj

    def ase_write_xyz(self, atom_obj):
        write(self.xyz_path, atom_obj, append=True)

    def run_opt(self, ml_engine):
        '''Reference: https://wiki.fysik.dtu.dk/ase/ase/optimize.html
        '''
        if self.geom_opt_algorithm == 'bfgs':
            from ase.optimize import BFGS
            optimizer = BFGS
        elif self.geom_opt_algorithm == 'lbfgs':
            from ase.optimize import LBFGS
            optimizer = LBFGS
        elif self.geom_opt_algorithm == 'gpmin':
            from ase.optimize import GPMin
            optimizer = GPMin
        else:
            raise(NotImplementedError)

        _atoms = self.ase_read_xyz()
        if ml_engine.lower() == 'deepmd':
            from deepmd.calculator import DP
            _atoms.calc = DP(model=self.pes_file_path)
        opt = optimizer(_atoms)
        opt.run() # probably need to set 'fmax'

        self.ase_write_xyz(_atoms)
        self.geom_opt = _atoms.get_positions()
        self.ene_opt  = _atoms.get_potential.energy()


        