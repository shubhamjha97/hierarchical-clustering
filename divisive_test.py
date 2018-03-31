import numpy as np
from data_reader import DataReader
from pathlib import Path
from timing_wrapper import timeit
from load_dist_matrix import load_dist_matrix
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import copy

class DivisiveClustering:
	'''
	Class which implements the DIANA (Divisive Analysis) algorithm and provides an interface for using the implementation.
	'''
	def __init__(self):
		'''
		Initializer for the DivisiveClustering class.
		This function initializes some important class variables.
		'''
		self.clusters={}
		self.dist_matrix=None
		self.mapping=None
		self.no_clusters=0
		self.linkage_matrix=None
		self.n=None
		self.hierarchical_clusters={}
		self.last_index=None

	def initialize(self):
		'''
		Initializer for the DivisiveClustering class.

		Parameters
		----------
		path : string
		    path to file containinng transactions.
		
		Returns
		----------
		transactions : list
			list containing all transactions. Each transaction is a list of
			items present in that transaction.
		items : list
			list containing all the unique items.
		'''
		self.n=len(self.mapping)
		self.last_index=2*self.n-2
		self.i=0
		self.clusters[self.last_index]=list(self.mapping.keys())
		self.linkage_matrix=np.zeros([self.n-1, 4])

	def splinter(self):
		'''
		This function finds the cluster with the largest diameter and finds the splinter element.

		Returns
		----------
		splinter_element : list
			list containing all transactions. Each transaction is a list of
			items present in that transaction.
		max_diameter_cluster : int
			index of the cluster with the largest diameter. This is the cluster which will be split.
		'''
		cluster_diameters={k:(len(v)<1)*(-1)+(len(v)>1)*np.max(self.dist_matrix[np.ix_(v,v)]) for k,v in self.clusters.items()}
		max_diameter_cluster=max(cluster_diameters, key=cluster_diameters.get)
		avg_within_cluster_distances={pt:(np.sum(self.dist_matrix[np.ix_(self.clusters[max_diameter_cluster], [pt])])/(len(self.clusters[max_diameter_cluster])-1)) for pt in self.clusters[max_diameter_cluster]}
		splinter_element=max(avg_within_cluster_distances, key=avg_within_cluster_distances.get)
		self.no_clusters+=1
		# print(cluster_diameters, max_diameter_cluster, splinter_element)
		splinter_element_dist=max(avg_within_cluster_distances)  ###### NOT NEEDED
		return splinter_element, max_diameter_cluster, splinter_element_dist

	def reassign(self, splinter_element, orig_cluster_key, splinter_element_dist):
		'''
		Initializer for the DivisiveClustering class.

		Parameters
		----------
		path : string
		    path to file containinng transactions.
		
		Returns
		----------
		transactions : list
			list containing all transactions. Each transaction is a list of
			items present in that transaction.
		items : list
			list containing all the unique items.
		'''
		# Create temp clusters
		temp_new_cluster=[splinter_element]
		self.clusters[orig_cluster_key].remove(splinter_element)
		temp_orig_cluster=self.clusters[orig_cluster_key]

		# Remove orig cluster from cluster dict
		del self.clusters[orig_cluster_key]

		# Calculate distances
		within_cluster_dist={pt:np.mean(self.dist_matrix[np.ix_(temp_orig_cluster,[pt])]) for pt in temp_orig_cluster }
		######### CHECK
		dist_to_splinter={pt:self.dist_matrix[pt, splinter_element]  for pt in temp_orig_cluster}
		dist_diff={pt:(within_cluster_dist[pt] - dist_to_splinter[pt]) for pt in temp_orig_cluster} # if +ve, move to splinter
		
		# Reassign points
		for pt in temp_orig_cluster:
			if dist_diff[pt]>=0 and len(temp_orig_cluster)>1:
				temp_new_cluster.append(pt)
				temp_orig_cluster.remove(pt)

		dist_bw_clusters=np.mean(self.dist_matrix[np.ix_(temp_orig_cluster, temp_new_cluster)])
		
		# Add temp clusters to cluster dict
		if len(temp_orig_cluster)==1:
			self.clusters[temp_orig_cluster[0]]=temp_orig_cluster
			orig_cluster_key=temp_orig_cluster[0]
		else:
			self.last_index-=1
			self.clusters[self.last_index]=temp_orig_cluster
			orig_cluster_key=self.last_index

		if len(temp_new_cluster)==1:
			self.clusters[temp_new_cluster[0]]=temp_new_cluster
			new_cluster_key=temp_new_cluster[0]
		else:
			self.last_index-=1
			self.clusters[self.last_index]=temp_new_cluster
			new_cluster_key=self.last_index

		# Append to hierarchical clusters
		self.hierarchical_clusters['iter_'+str(self.no_clusters)]=copy.deepcopy(self.clusters)

		# Make the linkage function ############# DIST ###############
		self.make_linkage_function(new_cluster_key, orig_cluster_key, dist_bw_clusters, len(temp_new_cluster)+len(temp_orig_cluster))

	def make_linkage_function(self, cluster_1, cluster_2, dist, len_cluster_2):
		'''
		Initializer for the DivisiveClustering class.

		Parameters
		----------
		path : string
		    path to file containinng transactions.
		
		Returns
		----------
		transactions : list
			list containing all transactions. Each transaction is a list of
			items present in that transaction.
		items : list
			list containing all the unique items.
		'''
		print(cluster_1, cluster_2, dist)
		self.linkage_matrix[self.n-self.no_clusters-1, 0]=cluster_2
		self.linkage_matrix[self.n-self.no_clusters-1, 1]=cluster_1
		self.linkage_matrix[self.n-self.no_clusters-1, 2]=dist
		self.linkage_matrix[self.n-self.no_clusters-1, 3]=len_cluster_2

	# def sanity_check_linkage(self):
	# 	for i in range(self.linkage_matrix.shape[0]):
	# 		if self.linkage_matrix[i, 0] >= self.n + i or self.linkage_matrix[i, 1] >=self. n + i:
	# 			print(i, self.linkage_matrix[i,:])

	def termination(self):
		'''
		Initializer for the DivisiveClustering class.

		Parameters
		----------
		path : string
		    path to file containinng transactions.
		
		Returns
		----------
		transactions : list
			list containing all transactions. Each transaction is a list of
			items present in that transaction.
		items : list
			list containing all the unique items.
		'''
		for k, v in self.clusters.items():
			if len(v)>1:
				return 0
		return 1

	def fit(self, dist_matrix, mapping):
		'''
		Initializer for the DivisiveClustering class.

		Parameters
		----------
		path : string
		    path to file containinng transactions.
		
		Returns
		----------
		transactions : list
			list containing all transactions. Each transaction is a list of
			items present in that transaction.
		items : list
			list containing all the unique items.
		'''
		self.dist_matrix=dist_matrix
		self.mapping=mapping
		self.initialize()
		while not self.termination():
			splinter_element, orig_cluster_key, splinter_element_dist=self.splinter()
			self.reassign(splinter_element, orig_cluster_key, splinter_element_dist)
		print('Clustering done!')
		self.sanity_check_linkage()

	def create_dendrogram(self):
		'''
		Initializer for the DivisiveClustering class.

		Parameters
		----------
		path : string
		    path to file containinng transactions.
		
		Returns
		----------
		transactions : list
			list containing all transactions. Each transaction is a list of
			items present in that transaction.
		items : list
			list containing all the unique items.
		'''
		np.savetxt('temp_matrix', self.linkage_matrix)
		fig=plt.figure()
		dendrogram(self.linkage_matrix, orientation='top')
		plt.show()
		fig.savefig('dendrogram.png')

def read_data():
	'''
	Initializer for the DivisiveClustering class.

	Parameters
	----------
	path : string
	    path to file containinng transactions.
	
	Returns
	----------
	transactions : list
		list containing all transactions. Each transaction is a list of
		items present in that transaction.
	items : list
		list containing all the unique items.
	'''
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
	dist_matrix=load_dist_matrix(data, mapping)

	model=DivisiveClustering()
	model.fit(dist_matrix, mapping)
	model.create_dendrogram()