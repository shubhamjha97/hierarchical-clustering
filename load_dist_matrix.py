import numpy as np
# from data_reader import DataReader
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler
from multiprocessing.pool import ThreadPool
import itertools
from similarity import similarity
from timing_wrapper import timeit
from time import time

count=0
def load_dist_matrix(data=None, k=None):
	pickleFilePath = Path('data/simMat_3.pkl')
	if pickleFilePath.is_file():
		temp_file=open(pickleFilePath, 'rb')
		return pickle.load(temp_file)
	if True:
		ClusterCount=len(data)
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

if __name__=='__main__':
	dist_matrix= load_dist_matrix()
	print(dist_matrix.shape)