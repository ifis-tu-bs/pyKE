def l1(vectors):
	'''
		Implements the l1 norm on a vectorspace.

			Parameters
		vectors - Tensor of dimension at least one,
		returning vectors whose norm is to be computed.

			Return Value
		Tensor of reduced dimension returning the norms.
		The order is preserved.
	'''

	from tensorflow import reduce_sum as sum
	return sum(abs(vectors), -1)


def l2(vectors):
	'''
		Implements the euclidean norm on a vectorspace.

			Parameters
		vectors - Tensor of dimension at least one,
		returning vectors whose norm is to be computed.

			Return Value
		Tensor of reduced dimension returning the norms.
		The order is preserved.
	'''

	from tensorflow import reduce_sum as sum, sqrt
	return sqrt(sum(vectors ** 2, -1))
