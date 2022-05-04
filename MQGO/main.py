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
		else:
			self.xyz_path = xyz_path

		self.qcsetting = qcsetting

		if consistensy_tol is None: 
			# consistensy_tol = 0.00045 * HARTREE2EV / BOHR # 0.00045 is from Gaussian
			consistensy_tol = 0.1 # eV
			logger.warning('No convergence tolarence is specified, default tolarence %12.11f will be used'%consistensy_tol)
			self.consistensy_tol = consistensy_tol
		else:
			self.consistensy_tol = consistensy_tol

		self.opt_algorithm = None
		self.opt_conv      = None
		self.global_temp   = None
		self.max_opt_cycle = None
		self.engine_path   = None
		
		self.debug = False

		self.have_init_qc_data = False
		self.target_geom = 1 # geometry to be optimized
        
	def check_consistensy(self, ML_opt_ene, QC_opt_ene):
		consistensy_met = np.abs((QC_opt_ene - ML_opt_ene)) < self.consistensy_tol
		# debug
		if self.debug:
			with open(self.work_path+'debug.txt', 'a+') as d:
				d.write('%20.12f    %20.12f    %20.12f    consistent?: %s\n'%(QC_opt_ene, ML_opt_ene, QC_opt_ene - ML_opt_ene, consistensy_met))
		return consistensy_met

	def kernel(self):
		# initialization
		if self.max_opt_cycle is None:
			self.max_opt_cycle = 100
		if self.opt_conv is None:
			self.opt_conv = 0.00045 #* HARTREE2EV / BOHR # 0.00045 is from Gaussian
		if self.target_geom is None:
			self.target_geom = 1 # optimize the first geometry as default

		if self.have_init_qc_data is True:
			logger.info('Initial QC data exists, preparation of initial QC data will be skipped')	 
			QC = QC_engine.QCEngine(qc_engine=self.qc_engine, xyz_path=self.xyz_path, geom_index= 1, **self.qcsetting).build()	
		else:
			# prepare initail QC refernce data
			# get basic information of the input xyz file
			atom_num, atom_symbol, coords = data.read_xyz(self.xyz_path, index=0, output='regular')
			geom_num = len(atom_num)
			append = False

			for index in range(0, geom_num):  # need to be finalized
				QC = QC_engine.QCEngine(qc_engine=self.qc_engine, xyz_path=self.xyz_path, geom_index=index + 1, **self.qcsetting).build()	
				E_QC, G_QC = QC.calc_new()

				# prepare input files for ml engine
				data1 = data.Data(self.qc_engine, self.ml_engine, self.work_path).build(QC)
				data1.dump(append=append)
				append = True
		
			QC.update_coord(coords[self.target_geom - 1] / BOHR)
		

		# loop starts here
		consistensy = False
		iter = 1
		while (not consistensy) and (iter < self.max_opt_cycle):
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
			
			Opt.conv_tol    = self.opt_conv
			Opt.global_temp = self.global_temp
			Opt.run_opt(self.ml_engine)
			E_ML = Opt.ene_opt
			F_ML = Opt.force_opt
			coord_ml_opt = Opt.geom_opt

			if self.debug:
				with open('./debug_ml_opt_force.txt','a+') as d:
					d.write('%24.18f\n'%E_ML)
					for i in range(0, len(F_ML)):
						for j in range(0, 3):
							d.write('%24.18f'%F_ML[i][j])
						d.write('\n')
					d.write('\n')

				with open('./debug_opt_geoms.txt','a+') as d:
					for i in range(0, len(coord_ml_opt)):
						for j in range(0, 3):
							d.write('%24.18f'%coord_ml_opt[i][j])
						d.write('\n')
					d.write('\n')

			# print('main.py 92:', coord_ml_opt)
			QC.update_coord(coord_ml_opt)
			# print('main.py 94: QC_coords1', QC.mol.atom_coords())
			E_QC, G_QC = QC.calc_new()
			# print('main.py 96:', E_QC)

			consistensy = self.check_consistensy(E_ML, E_QC * HARTREE2EV)
			data1 = data.Data(self.qc_engine, self.ml_engine, self.work_path).build(QC)
			data1.dump(append=append)
			iter += 1
			# break

		self.opt_geom = coord_ml_opt

		if self.debug:
			with open('./debug_ml_opt_force.txt','a+') as d:
				d.write('********************\n')
			with open('./debug_opt_geoms.txt','a+') as d:
				d.write('********************\n')
