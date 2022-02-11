#'''This code is the main frame of the interface.'''
from QC_engine import QCEngine
import numpy as np

class MLgeomopt(self):
	def __init__(self):
		if self.consistensy_tol is None:
			self.consistensy_tol = 0.000001
	def check_consistensy(self, ML_opt_ene, QC_opt_ene):
		consistensy_met = np.abs((QC_opt_ene - ML_opt_ene)) < self.consistensy
		return consistensy_met

	def kernel(self):
		E_QC, G_QC = QCEngine(molecule, engine).calc_new()

		consistensy = False
		iter = 0
		while consistensy is not True and iter < self.max_cycle:
			MLEngine().add_geom()
			MLEngine().training()
			E_ML = MLEngine().calc_new()
			consistensy = self.check_consistensy(E_ML, E_QC)
			iter += 1