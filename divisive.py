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
import line_profiler
from load_sim_matrix import load_similarity_matrix
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import copy


class DivisiveClustering:
	def __init__(self):
		self.clusters={}
		self.sim_matrix=None
		self.mapping=None
		self.no_clusters=0
		self.linkage_matrix=None
		self.n=None
		self.i=None
		self.hierarchical_clusters={}

	def splinter(self):
		cluster_diameters={k:(len(v)<1)*(-1)+(len(v)>1)*np.max(np.max(self.sim_matrix[np.ix_(v,v)], axis=1)) for k,v in self.clusters.items()}
		max_diameter_cluster=max(cluster_diameters, key=cluster_diameters.get)
		avg_within_cluster_distances={pt:np.mean(self.sim_matrix[:, pt]) for pt in self.clusters[max_diameter_cluster]}
		splinter_element=max(avg_within_cluster_distances, key=avg_within_cluster_distances.get)
		self.clusters[max_diameter_cluster].remove(splinter_element)
		self.no_clusters+=1
		self.clusters[self.no_clusters]=[splinter_element]
		return self.no_clusters, max_diameter_cluster

	def initialize(self):
		self.clusters[0]=list(self.mapping.keys())
		self.n=len(self.mapping)
		self.i=0
		self.linkage_matrix=np.zeros([self.n-1, 4])

	def reassign(self, new_cluster_key, orig_cluster_key):
		splinter_element=self.clusters[new_cluster_key][0]
		within_cluster_dist={pt:np.mean(np.delete(self.sim_matrix[:, pt], (pt, splinter_element))) for pt in self.clusters[orig_cluster_key]}
		dist_to_splinter={pt:self.sim_matrix[pt, splinter_element]  for pt in self.clusters[orig_cluster_key]}
		dist_diff={pt:(within_cluster_dist[pt] - dist_to_splinter[pt]) for pt in self.clusters[orig_cluster_key]} # if +ve, move to splinter
		for pt in self.clusters[orig_cluster_key]:
			if dist_diff[pt]>0 and len(self.clusters[orig_cluster_key])>1:
				self.clusters[new_cluster_key].append(pt)
				self.clusters[orig_cluster_key].remove(pt)
		self.hierarchical_clusters[self.no_clusters]=copy.deepcopy(self.clusters)
		new_cluster_elements=self.clusters[new_cluster_key]
		orig_cluster_elements=self.clusters[orig_cluster_key]
		print(orig_cluster_key, new_cluster_key)
		temp_cluster_size=len(self.clusters[new_cluster_key])
		if len(new_cluster_elements)==1:
			new_cluster_key=self.no_clusters
			orig_cluster_key=self.n+self.no_clusters
		elif len(orig_cluster_elements)==1:
			orig_cluster_key=self.no_clusters
			new_cluster_key=self.n+self.no_clusters
		print(orig_cluster_key, new_cluster_key)
		self.make_linkage_function(new_cluster_key, orig_cluster_key, 0, temp_cluster_size)


	def make_linkage_function(self, cluster_1, cluster_2, dist, len_cluster_2):
		# print(cluster_1, cluster_2, dist, len_cluster_2)
		self.linkage_matrix[self.no_clusters-1, 0]=cluster_1
		self.linkage_matrix[self.no_clusters-1, 1]=cluster_2
		# self.linkage_matrix[2*self.n-self.no_clusters-1, 2]=dist
		# self.linkage_matrix[2*self.n-self.no_clusters-1, 3]=len_cluster_2

	def termination(self):
		for k, v in self.clusters.items():
			if len(v)>1:
				return 0
		return 1

	def fit(self, sim_matrix, mapping):
		self.sim_matrix=sim_matrix
		self.mapping=mapping
		self.initialize()
		while not self.termination():
			new_cluster_key, orig_cluster_key=self.splinter()
			self.reassign(new_cluster_key, orig_cluster_key)
		print('Clustering done!')

	def create_dendrogram(self):
		fig=plt.figure()
		#self.linkage_matrix=self.linkage_matrix[self.linkage_matrix[:,3].argsort()]
		print(self.linkage_matrix)
		dendrogram(self.linkage_matrix, orientation='top')
		plt.show()
		fig.savefig('dendrogram.png')

@timeit
def read_data():
	dataDict={}
	rawData=None
	with open('data/data.txt', 'r') as f:
		rawData=f.read()
	dataSegments = rawData.split('>')
	for segment in dataSegments:
		try:
			lines = segment.split('\n', 1)
			dataDict[lines[0]] = lines[1].replace('\n', '')
		except:
			pass
	mapping={ind:k for ind, k in enumerate(dataDict.keys())}
	return dataDict, mapping

if __name__=='__main__':
	data, mapping=read_data()
	sim_matrix=np.random.random(size=[311,311])#load_similarity_matrix(data, mapping)

	model=DivisiveClustering()
	model.fit(sim_matrix, mapping)
	print(model.hierarchical_clusters)
	model.create_dendrogram()