import numpy as np
''' Utiltiy function for creating the Distance between DNA sequences
	based on their edit distance using preset penalties '''
def computeDistance(str1, str2, scoring_fn=None):
	if not scoring_fn:
		MATCH_SCORE=0
		MISMATCH_PENALTY=1
		GAP_PENALTY=2
	score_mat=np.zeros([len(str1)+1, len(str2)+1])
	for i in range(len(str1)+1):
		for j in range(len(str2)+1):
			if(i==0 or j==0):
				entry=i*GAP_PENALTY + j*GAP_PENALTY
			else:
				entry=min((score_mat[i, j-1]+GAP_PENALTY), (score_mat[i-1, j]+GAP_PENALTY),
					score_mat[i-1, j-1]+(MATCH_SCORE*int(str1[i-1]==str2[j-1])+MISMATCH_PENALTY*(str1[i-1]!=str2[j-1])))
			score_mat[i, j]=entry
	score=score_mat[len(str1), len(str2)]
	return score

if __name__=='__main__':
	for i in range(1):
			score=computeSimilarity('''ACGT''', '''ACGTA''')
			print(score)