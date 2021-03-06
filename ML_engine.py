from re import sub
from tkinter import W
import numpy as np
import subprocess


class MLEngine(object):
    def __init__(self, work_path, ml_engine):
        self.work_path = work_path # work_path is the path of raw files for ML engine
        if ml_engine.lower()== 'deepmd':
            self.engine = DeePMD

    def build(self):
        return self.engine(work_path=self.work_path)

class DeePMD(object):
    def __init__(self, work_path=None):
        assert work_path is not None, 'Please specify the working directory.'
        self.work_path = work_path
        # check input 

    def training(self):
        train = subprocess.run('dp train input.jsoon', shell=True)
        assert train.returncode == 0, 'An error occurred during training process.'
    
    def freeze(self):
        freeze = subprocess.run('dp freeze -o graph.pb', shell=True)
        assert freeze.returncode == 0, 'An error occurred during freezing process.'

    def run(self):
        self.training()
        self.freeze()


        





