import subprocess

class MLEngine():
    def __init__(self, work_path=None, ml_engine=None):
        assert ml_engine is not None, 'Please specify which QC engine to use.'
        assert work_path is not None, 'Please specify the path of xyz file.'
        
        self.work_path = work_path # work_path is the path of raw files for ML engine
        if ml_engine.lower()== 'deepmd':
            self.engine = DeePMD

    def build(self):
        return self.engine(work_path=self.work_path)

class DeePMD():
    def __init__(self, work_path=None):
        assert work_path is not None, 'Please specify the working directory.'
        self.work_path  = work_path
        self.raw_path   = work_path+'raw/'
        self.input_path = work_path+'input.json'

    def training(self):
        train = subprocess.run('dp train %s'%self.input_path, shell=True)
        assert train.returncode == 0, 'An error occurred during training process.'
    
    def freeze(self):
        freeze = subprocess.run('dp freeze -o graph.pb', shell=True)
        assert freeze.returncode == 0, 'An error occurred during freezing process.'

    def compress(self):
        compress = subprocess.run('dp compress -i graph.pb -o graph-compress.pb', shell=True)
        assert compress.returncode == 0, 'An error occurred during compressing process.'

    def clear_temp(self):
        subprocess.run('rm model.ckpt*; rm checkpoint; rm -r model-compression', shell=True)

    def run(self):

        self.training()
        self.freeze()
        self.compress()
        self.clear_temp()

