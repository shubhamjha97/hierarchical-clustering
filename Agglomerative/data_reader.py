import pickle
import random
from pathlib import Path

class DataReader:
	''' Data Reader Class for Parsing the DNA data and saving it '''
	def __init__(self):
		''' Initialize default variables '''
		self.dataFolder = Path('data/')
		self.dataRead = Path('data.txt')
		self.dataWrite = Path('data/data.pkl')
		self.sample = Path('data/sampledata.txt')
		self.dataDict = dict()
		self.dataArray = []
		self.rawData = ''
		self.checkFolder()

	def checkFolder(self):
		''' Check if folder exists. If not create it'''
		if self.dataFolder.is_dir():
			pass
		else:
			self.dataFolder.mkdir(exist_ok=True, parents=True)

	def run(self):
		''' Execute reader functions '''
		self.readData()
		self.createDict()

	def readData(self):
		''' Read the Data from raw text file '''
		with open(self.dataRead, 'r+', encoding='utf-8') as file:
			self.rawData = file.read()

	def createDict(self):
		''' create the dictionary for the DNA Sequences '''
		self.dataSegments = self.rawData.split('>')
		for segment in self.dataSegments:
			try:
				lines = segment.split('\n', 1)
				self.dataDict[lines[0]] = lines[1].replace('\n', '')
			except:
				pass
		self.saveData()

	def saveData(self):
		''' Save this dictionary into a pickle file for future use '''
		with open(self.dataWrite, 'wb') as file:
			pickle.dump(self.dataDict, file)

	def dataSample(self):
		''' create a sample data and write it to a file '''
		randomKey = random.choice(list(self.dataDict.keys()))
		with open(self.sample, 'w') as file:
			file.write(randomKey + ' : ' + self.dataDict[randomKey] +'\n' + str(len(self.dataDict[randomKey])))

	def loadData(self):
		''' load data from pickle file. If pickle file does not exists, call readData and createDict first '''
		if self.dataWrite.exists():
			with open(self.dataWrite, 'rb') as file:
				self.dataDict = pickle.load(file)
		else:
			self.run()
		self.dataArray = [self.dataDict[k] for k in self.dataDict.keys()]
		# print(str(len(self.dataDict)))
		return self.dataDict

	def getDataArray(self):
		''' Return the array containing just the DNA Sequences '''
		return self.dataArray

if __name__ == "__main__":
	d = DataReader()
	d.loadData()
	d.dataSample()
	# print(d.getDataArray())

