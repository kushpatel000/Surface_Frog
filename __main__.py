import sys
import os
import importlib
import json
import time
import numpy as np
from engine import engine

from multiprocessing import Pool
from functools import partial


def run_instance( paramFile,pidx ):
	the_matrix = engine( paramFile,pidx )
	# print("starting ",pidx)
	res = the_matrix.simulate()
	return res

if __name__ == '__main__':
	
	if len(sys.argv) < 2:
		print("No parameter file.")
		exit(0)
	paramFile = sys.argv[0]
	if (".py" in paramFile):
		paramFile = sys.argv[1]


	# import parameters
	with open(paramFile) as json_file:
		data = json.load(json_file)
		mdl_nm	= data['model']  if 'model'	in data else 'SimpleAvoidedCrossing'
		nsims	= data['nsims']  if 'nsims'	in data else 1
		nprocs	= data['nprocs'] if 'nprocs'in data else 1
		tag		= data['tag']	 if 'tag'	in data else 'sim'
		logQ	= data['logQ']	 if 'logQ'	in data else False
	
	# Import the requested model to see if it exists
	try:
		imp_mod = importlib.import_module('models.'+mdl_nm)
		mdl     = getattr( imp_mod,mdl_nm )
	except:
		mdl     = None
	if not mdl:
		print("Model not found. Exiting...")
		exit(0)
	del mdl

	# If logging requested, make the directory 
	if logQ and not os.path.isdir("LogFiles"):
		os.mkdir("LogFiles")


	# Begin Simulations
	start_time = time.time()
	total = np.zeros((2,2))
	if nprocs == 1: 
		# Don't parallel
		for k in range(nsims):
			the_matrix = engine( paramFile,k )
			total += the_matrix.simulate()
	else:
		# Parallel
		ids = [k for k in range(nsims)]
		preprocessed = partial( run_instance, paramFile  )
		pool = Pool(processes=nprocs)
		res = pool.map( preprocessed, ids )
		
		pool.close()
		pool.join()
		total = sum(res)
	end_time = time.time()


	if not os.path.isdir("Output"): os.mkdir("Output")
	# Output
	outfile = "Output/"+mdl_nm+"."+tag 
	with open(outfile,'w+') as wrtr:
		wrtr.write( "{} simulations over {} processes finished in {}.\n"
			.format(nsims,nprocs,end_time-start_time) )
		for s in range(total.shape[0]):
			wrtr.write("State {0:2d} - Reflection:   {1:4d}\n"
				.format(s, int(total[s,0])))
			wrtr.write("State {0:2d} - Transmission: {1:4d}\n"
				.format(s, int(total[s,1])))
