import subprocess

class MLEngine(object):
    def __init__(self, work_path=None, ml_engine=None, engine_path=None):
        assert ml_engine is not None, 'Please specify which QC engine to use.'
        assert work_path is not None, 'Please specify the path of xyz file.'
        
        self.work_path = work_path # work_path is the path of raw files for ML engine
        if ml_engine.lower()== 'deepmd':
            self.engine = DeePMD
            self.engine_path = engine_path

    def build(self):
        return self.engine(work_path=self.work_path, engine_path=self.engine_path)

class DeePMD(object):
    def __init__(self, work_path=None, engine_path=None):
        assert work_path is not None, 'Please specify the working directory.'

        self.work_path = work_path
        if engine_path is None:
            engine_path = '~/deepmd-kit/'
        self.engine_path = engine_path

    def make_npy(self, frame=None):
        assert frame is not None, 'Please specify the number of frames for a system'
        print("ML_engine.py 27: frame", frame)
        make = subprocess.run('%s'%self.engine_path+'data/raw/raw_to_set.sh %d'%frame, shell=True)
        assert make.returncode == 0, 'An error occurred during preparing npy file process.'

    def training(self):
        train = subprocess.run('dp train input.json', shell=True)
        assert train.returncode == 0, 'An error occurred during training process.'
    
    def freeze(self):
        freeze = subprocess.run('dp freeze -o graph.pb', shell=True)
        assert freeze.returncode == 0, 'An error occurred during freezing process.'

    def run(self, frame=None):
        self.make_npy(frame=frame)
        self.training()
        self.freeze()
