'''This code is the main frame of the interface.'''

import os
import logging

logger = logging.getLogger(__name__)

from MQGO import data, QC_engine, ML_engine, optimizer
from pyscf.data.nist import BOHR ,HARTREE2EV
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
			xyzs = [i for i in os.path.listdir(current_path) if 'xyz' in i]
			if len(xyzs) > 1:
				logger.warning('There are mutiple xyz files in the folder, %s will be set as default'%xyzs[0])
				self.xyz_paths = []
				for i in xyzs:
					self.xyz_paths.append(current_path + i)
			
			self.xyz_path = xyzs[0]
			# for i in os.path.listdir(current_path):
			# 	if 'xyz' in i:
					# self.xyz_path = current_path + i
		else:
			self.xyz_path = xyz_path

		self.qcsetting = qcsetting

		if consistensy_tol is None: 
			consistensy_tol = 0.00045 * HARTREE2EV / BOHR # 0.00045 is from Gaussian
			logger.warning('No convergence tolarence is specified, default tolarence %12.11f will be used'%consistensy_tol)
			self.consistensy_tol = consistensy_tol
		else:
			self.consistensy_tol = consistensy_tol

		self.opt_algorithm = None
		self.opt_conv      = None
		self.global_temp   = None
		self.max_opt_cycle = None
		self.engine_path   = None
        
	def check_consistensy(self, ML_opt_ene, QC_opt_ene):
		consistensy_met = np.abs((QC_opt_ene - ML_opt_ene)) < self.consistensy_tol
		# debug
		with open(self.work_path+'debug.txt', 'a+') as d:
			d.write('%20.12f    %20.12f    %20.12f    consistent?: %s\n'%(QC_opt_ene, ML_opt_ene, QC_opt_ene - ML_opt_ene, consistensy_met))
		return consistensy_met

	def kernel(self):
		# initialization
		if self.max_opt_cycle is None:
			self.max_opt_cycle = 100
		QC = QC_engine.QCEngine(qc_engine=self.qc_engine, xyz_path=self.xyz_path, **self.qcsetting).build()	
		E_QC, G_QC = QC.calc_new()
		
		append = False
		data1 = data.Data(self.qc_engine, self.ml_engine, self.work_path).build(QC)
		data1.dump(append=append)

		# loop starts here
		consistensy = False
		append = True
		iter = 1
		while consistensy is not True and iter < self.max_opt_cycle:
			print('''
			###########################################
			###      Optimization cycle: %5d      ###
			###########################################
			'''%iter)
			ML = ML_engine.MLEngine(work_path=self.work_path, ml_engine=self.ml_engine).build()
			ML.run()

			Opt = optimizer.Optimizer(xyz_path=self.xyz_path,
									work_path=self.work_path,
									ml_engine=self.ml_engine,
									outter_cycle=iter,
									algorithm=self.opt_algorithm,
									max_opt_cycle=self.max_opt_cycle)
			
			Opt.conv_tol    = self.consistensy_tol
			Opt.global_temp = self.global_temp
			Opt.run_opt(self.ml_engine)
			E_ML = Opt.ene_opt
			coord_ml_opt = Opt.geom_opt
			# print('main.py 92:', coord_ml_opt)
			QC.update_coord(coord_ml_opt)
			# print('main.py 94:', QC.mol.atom_coords())
			E_QC, G_QC = QC.calc_new()
			# print('main.py 96:', E_QC)

			consistensy = self.check_consistensy(E_ML, E_QC * HARTREE2EV)
			data1 = data.Data(self.qc_engine, self.ml_engine, self.work_path).build(QC)
			data1.dump(append=append)
			iter += 1
			# break


		self.opt_geom = coord_ml_opt
