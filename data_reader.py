import pickle
import random
from pathlib import Path

class DataReader:
	def __init__(self):
		self.dataFolder = Path('data/')
		self.dataRead = Path('data/data.txt')
		self.dataWrite = Path('data/data.pkl')
		self.sample = Path('data/sampledata.txt')
		self.dataDict = dict()
		self.dataArray = []
		self.rawData = ''
		self.checkFolder()

	def checkFolder(self):
		if self.dataFolder.is_dir():
			pass
		else:
			self.dataFolder.mkdir(exist_ok=True, parents=True)

	def run(self):
		self.readData()
		self.createDict()

	def readData(self):
		with open(self.dataRead, 'r+', encoding='utf-8') as file:
			self.rawData = file.read()

	def createDict(self):
		self.dataSegments = self.rawData.split('>')
		for segment in self.dataSegments:
			try:
				lines = segment.split('\n', 1)
				self.dataDict[lines[0]] = lines[1].replace('\n', '')
			except:
				pass
		self.saveData()

	def saveData(self):
		with open(self.dataWrite, 'wb') as file:
			pickle.dump(self.dataDict, file)

	def dataSample(self):
		randomKey = random.choice(list(self.dataDict.keys()))
		with open(self.sample, 'w') as file:
			file.write(randomKey + ' : ' + self.dataDict[randomKey] +'\n' + str(len(self.dataDict[randomKey])))

	def loadData(self):
		if self.dataWrite.exists():
			with open(self.dataWrite, 'rb') as file:
				self.dataDict = pickle.load(file)
		else:
			self.run()
		self.dataArray = [self.dataDict[k] for k in self.dataDict.keys()]
		# print(str(len(self.dataDict)))
		return self.dataDict

	def getDataArray(self):
		return self.dataArray

if __name__ == "__main__":
	d = DataReader()
	d.loadData()
	d.dataSample()
	# print(d.getDataArray())

