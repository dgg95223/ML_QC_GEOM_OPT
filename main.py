#'''This code is the main frame of the interface.'''
import subprocess
import logging
logger = logging.getLogger(__name__)

from QC_engine import QCEngine
from ML_engine import MLEngine
import numpy as np


class MLgeomopt():
	def __init__(self, qc_engine=None, ml_engine=None, work_path=None, xyz_path=None, consistensy_tol=None, **qcsetting):
		if qc_engine is None:
			logger.warning('No QC engine is specified, PySCF will be used.')
			self.qcengine = 'pyscf'
		else:
			self.qcengine = qc_engine
		
		if ml_engine is None:
			logger.warning('No ML engine is specified, DeePotential will used.')
			self.mlengine = 'deepmd'
		else:
			self.mlengine = ml_engine

		if work_path is None:
			logger.warning('No work path is specified, the current path will be used as the default work path')
			self.workpath = './'
		else:
			self.workpath = work_path

		if xyz_path is None:
			logger.warning('No path of xyz file is specified, the current path will be used as the default path.')
			current_path = './'
			self.xyz_path = subprocess.run('ls '+current_path+'*xyz', shell=True)
		else:
			self.xyz_path = xyz_path

		self.qcsetting = qcsetting

		if consistensy_tol is None:
			consistensy_tol = 1e-6
			logger.warning('No convergence tolarence is specified, default tolarence %12.11 will be used'%consistensy_tol)
			self.consistensy_tol = consistensy_tol
		else:
			self.consistensy_tol = consistensy_tol
		
        
	def check_consistensy(self, ML_opt_ene, QC_opt_ene):
		consistensy_met = np.abs((QC_opt_ene - ML_opt_ene)) < self.consistensy
		return consistensy_met

	def kernel(self, workpath, xyz_path, qcengine, mlengine):
		E_QC, G_QC = QCEngine(qc_engine=qcengine, xyz_path=xyz_path, **self.qcsetting).calc_new()


		consistensy = False
		iter = 0
		while consistensy is not True and iter < self.max_cycle:
			MLEngine(workpath, mlengine).add_geom()
			MLEngine(workpath, mlengine).training()
			E_ML = MLEngine().calc_new()
			consistensy = self.check_consistensy(E_ML, E_QC)
			iter += 1
