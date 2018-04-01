import numpy as np
from time import time
import line_profiler

def timer(func):
	def wrapper(*args):
		t = time()
		res=func(*args)
		print(time()-t)
		return res
	return wrapper

def cvt(s):
	s_clean=s.replace('\n','').replace('\t', '')
	conv_dict={'A':1, 'T':2, 'G':3, 'C':4}
	return [conv_dict[x] for x in s_clean]

def similarity(str1, str2):
	MATCH_SCORE, MISMATCH_PENALTY, GAP_PENALTY=(2, -1, -2)
	str1_len, str2_len=(len(str1), len(str2))
	score_mat=np.zeros([str1_len+1, str2_len+1])
	score_mat[0,:]=np.array([j*GAP_PENALTY for j in range(str2_len+1)])
	score_mat[1:,0]=np.array([i*GAP_PENALTY for i in range(1, str1_len+1)])
	for i in range(1, str1_len+1):
		for j in range(1, str2_len+1):
			i_1=i-1
			j_1=j-1
			is_match=(str1[i-1]==str2[j-1])
			score_mat[i,j]=max((score_mat[i, j_1]+GAP_PENALTY), (score_mat[i_1, j]+GAP_PENALTY), score_mat[i_1, j_1]+(MATCH_SCORE*(is_match)+MISMATCH_PENALTY*(not is_match)))
	return score_mat[str1_len, str2_len]

if __name__=='__main__':
		score=similarity(cvt('''ATGGCTCAGACAAGATATACACAAAATAGATGGAGAAATGAAGCTTGTCGAGAGAAAGCC
	CTTTCTACATGTGGTTGTTCAGCTAATGTGTCTCAACCCACAATTACAACATTGCTGACA
	CCATTAACCAGTGAAACAACACCACTTCGCGAAATCCTTGTTGTCTCATTAAAAAGGAAA
	GGGTCAGATGATGTAAGGCATGCAATCAAAGACAATAACACTCTCTGCCCATTTGTCATC
	TTAAAGGAGCCGATCAACGCGCCATCCCTCGTGTGCCATCTACATAAGAGTTGTTGCCGA
	CACAGGCAACTCCAGAGGAGTTTGCGCCTTAAAAACTACCTAGAGTGCTATACTTCATAG'''), cvt('''ATGGAAGAATTTATTGCCCAGAAAATTCCATTCTGCTATCTGATTCAAAAAGTCCAGTCC
	CCCCAGCTTCGGAAAATCTATTTTCCACATTTTAATACCCTGCAGAACAGTCCTCATAAC
	TCATCCGAGTGTGTTAAGCACAGTTTTATTAGATCTGAAACAAATTTTGATGTGGTGGCA
	GCGATAATTCCAATTTTATGCCTTTGTTATGGCTTTAATCACTTTCTTTATGCGTTTTGT
	TCCAGCGATCAGGGAGAGACACCTGATAGGCAAAGAGGATACCAGGGGAACCATTTTTAT
	TTGGAATGTGGTGAGAAGTCTAATTAG'''))
	# score=similarity('ATGC', 'ACCT')
	# print(score)