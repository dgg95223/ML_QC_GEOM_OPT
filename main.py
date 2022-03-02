#'''This code is the main frame of the interface.'''
from QC_engine import QCEngine
from ML_engine import MLEngine
import numpy as np

class MLgeomopt():
	def __init__(self, consistensy_tol=None, **qcsetting):
		if consistensy_tol is None:
			self.consistensy_tol = 0.000001
		else:
			self.consistensy_tol = consistensy_tol
		
		self.qcsetting = qcsetting
        
	def check_consistensy(self, ML_opt_ene, QC_opt_ene):
		consistensy_met = np.abs((QC_opt_ene - ML_opt_ene)) < self.consistensy
		return consistensy_met

	def kernel(self, workpath, xyz_path, qcengine, mlengine):
		E_QC, G_QC = QCEngine(xyz_path, qcengine, self.qcseting).calc_new()

		consistensy = False
		iter = 0
		while consistensy is not True and iter < self.max_cycle:
			MLEngine(workpath, mlengine).add_geom()
			MLEngine(workpath, mlengine).training()
			E_ML = MLEngine().calc_new()
			consistensy = self.check_consistensy(E_ML, E_QC)
			iter += 1
