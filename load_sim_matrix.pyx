import numpy as np
from data_reader import DataReader
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
from multiprocessing.pool import ThreadPool
import itertools
from similarity import similarity
from timing_wrapper import timeit
from time import time

count=0
def load_similarity_matrix(data=None, k=None):
	ClusterCount=len(data)
	pickleFilePath = Path('data/simMat_2.pkl')
	if True:
		count=0
		pool=ThreadPool()
		sim_matrix=np.ones([ClusterCount, ClusterCount])
		def fill_matrix(coord):
			start_time=time()
			global count
			val=similarity(data[k[coord[0]]], data[k[coord[1]]])
			count+=1
			sim_matrix[coord[0], coord[1]]=val
			print(count, count/(ClusterCount**2)*100, 'time:', time()-start_time)
		keys_list=list(itertools.product(range(ClusterCount), range(ClusterCount)))[0:100]
		pool.map(lambda x: fill_matrix(x), keys_list)
		with open('sim_matrix_div.pkl', 'wb') as file:
			pickle.dump(sim_matrix, file)