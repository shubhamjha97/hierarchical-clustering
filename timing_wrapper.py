from time import time

def timeit(fn):
	def wrapper(*args, **kwargs):
		start = time()
		res = fn(*args, **kwargs)
		print(fn.__name__, "took", time() - start, "seconds.")
		return res
	return wrapper