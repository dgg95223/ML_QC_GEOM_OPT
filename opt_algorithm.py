'''Geometry optimization module of ASE'''

from ase.io import read, write

class Optimizer(object):
    def __init__(self, mainobject, ml_engine, algorithm=None, opt_cycle):
        self.atom_symbol = mainobject.atom_symbol
        self.cycle = opt_cycle
        algorithms = ['bfgs', 'lbfgs', 'gpmin', 'pyberny'] 

        self.pes_file_path = ml_engine.file_path
        self.xyz_path = mainobject.xyz_path

        if algorithm is None:
            self.geom_opt_algorithm = 'bfgs'
        else:
            assert algorithm in algorithms, 'The selected algorithm is not supported with ASE optimizer.'
            self.geom_opt_algorithm = algorithm
        
        self.optimized_geom = None

    def read_xyz(self): # Build 'Atoms'object from xyz file, a xyz file may contain multiple geometries
        ''' ASE's Atoms object usage: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
        For current implementation, each geometry is read/written in a single xyz file, an aternative way
        is to write all geometries in one file.
        '''
        
        atoms_obj = read(self.xyz_path, index=self.cycle)
        return atoms_obj

    def write_xyz(self, atom_obj):
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

        _atoms = self.read_xyz()
        if ml_engine.lower() == 'deepmd':
            from deepmd.calculator import DP
            _atoms.calc = DP(model=self.pes_file_path)
        opt = optimizer(_atoms)
        opt.run() # probably need to set 'fmax'

        self.write_xyz(_atoms)
        self.optimized_geom = _atoms.geit_positions()
        

    


        

dyn = BFGS(water)
dyn.run(fmax=1e-6)
print(water.get_positions())