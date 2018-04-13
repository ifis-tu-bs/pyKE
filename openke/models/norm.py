from tensorflow import reduce_sum as sum, reduce_min as min, reduce_max as max


def l1(vectors):
	'''Implements the l1 norm on a vectorspace.'''
	return sum(abs(vectors), -1)


def l2(vectors):
	'''Implements the euclidean norm on a vectorspace.'''
	return sqrt(sum(vectors ** 2, -1))
