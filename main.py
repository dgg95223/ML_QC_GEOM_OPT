#'''This code is the main frame of the interface.'''

import subprocess
import logging
logger = logging.getLogger(__name__)

import io, QC_engine, ML_engine
import numpy as np


class MLgeomopt():
	def __init__(self, qc_engine=None, ml_engine=None, work_path=None, xyz_path=None, consistensy_tol=None, **qcsetting):
		if qc_engine is None:
			logger.warning('No QC engine is specified, PySCF will be used.')
			self.qc_engine = 'pyscf'
		else:
			self.qc_engine = qc_engine
		
		if ml_engine is None:
			logger.warning('No ML engine is specified, DeePotential will used.')
			self.ml_engine = 'deepmd'
		else:
			self.ml_engine = ml_engine

		if work_path is None:
			logger.warning('No work path is specified, the current path will be used as the default work path')
			self.work_path = './'
		else:
			self.work_path = work_path

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

	def kernel(self):
		# initialization
		QC = QC_engine.QCEngine(qc_engine=self.qc_engine, xyz_path=self.xyz_path, **self.qcsetting)
		E_QC, G_QC = QC.calc_new()
		
		append = False
		data = io.Data(self.qcengine, self.ml_engine, self.work_path).build(QC)
		data.dump(append=append)

		# loop starts here
		consistensy = False
		iter = 0
		while consistensy is not True and iter < self.max_cycle:
			E_QC, G_QC = QC.calc_new()
		
			

			ML = ML_engine.MLEngine(self.work_path, self.ml_engine)

			ML.training()
			E_ML = ML_engine.MLEngine().calc_new()
			consistensy = self.check_consistensy(E_ML, E_QC)
			iter += 1
