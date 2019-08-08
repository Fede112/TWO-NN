import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_data(file_path):
	"""
	load data and return an nd numpy array
	"""
	data = np.loadtxt(file_path, skiprows = 0, unpack = False)
	return data



def load_data_pkl(file_path):
	with open(input_file, 'rb') as file:
		return np.load(file_path)


def dist_mat(data, metric):
	"""
	Computes the distance matrix of a dataset given a specific metric function.
	The diagonal is set to float("inf")
	:param data: dataset as a 2d numpy array.
	:param metric: function that defines the metric.
	"""
	N_samples = data.shape[0]
	N_coord = data.shape[1]
	dist_mat = np.zeros((N_samples, N_samples))
	for i in range(N_samples):
		dist_mat[i,i] = float("inf")
		for j in range(i+1,N_samples):
			dist_mat[i,j] = metric(data[i,:], data[j,:])
			dist_mat[j,i] = dist_mat[i,j]
	return dist_mat


def arr_two_minval(array):
	"""
	Returns the two minimm values of an array.
	:param array: list or array of values.
	"""
	# RFI: maybe this implementation is faster: 
	# first, second = sorted(x)[:2] 
	first = float("inf")
	second = float("inf")
	for val in array:
		if val < first:
			second = first
			first = val
		elif val < second and second != first:
			second = val
	return first, second



def calc_two_nn(dist_mat2):
	"""
	Takes as input a squared distance matrix and returns the two
	minimum values per row.
	:param dist_mat2: distance squared matrix between data points. It uses the squared
						distance matrix ( A[i,j] = d(i,j)**2 ) to improve performance.
	"""
	r1_arr = []
	r2_arr = []
	for row in dist_mat2:
		first, second = arr_two_minval(row)
		r1_arr.append(np.sqrt(first))
		r2_arr.append(np.sqrt(second))
	return np.array(r1_arr), np.array(r2_arr)



def two_nn_id(r1_arr, r2_arr, frac = .9):
	"""
	Computes the intrinsic dimension of a given dataset using the TWO-NN algorithm.
	:param data: dataset as a 2d numpy array.
	:param frac: fraction of points used to calculate the slope of the linear regression. \
	Not to be confused with the fraction of data used in the block analysis.
	"""
	assert r1_arr.shape[0] == r2_arr.shape[0], 'length of r1_arr is not the same as r2_arr'

	N_samples = r1_arr.shape[0]
	# print(f"N_samples: {N_samples}")
	N_frac = int(N_samples*frac)
	print(f'N_frac: {N_frac}')
	if N_frac == 0:
		N_frac = 1
	assert N_frac <= N_samples and N_frac > 0, 'Frac must be a real number between 0 and 1.'

	mu = r2_arr/r1_arr

	# mu is sorted
	mu, mu_cs = cum_sum(mu)
	# mu = np.sort(mu)

	# linearize the equation F(mu) = (1-mu**(-d))
	y = - np.log(1-mu_cs)[:N_frac]
	x = np.log(mu)[:N_frac]
	print(f'len x: {len(x)}')

	# Linear regression with intercept = 0
	dim = np.sum(x*y) / np.sum(x*x)
	# alternative using numpy
	# x = x[:,np.newaxis]
	# dim, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

	print(dim)
	return dim, x, y


def eucl_dist2(x,y):
	""" Computes the euclidean distance between two points"""
	return sum((x - y)**2)


def cum_sum(x):
	"""
	Calculates the cumulative sum of an array x.
	:param x: array with values.
	"""
	x_sort = np.sort(x)
	x_len = len(x)
	# calculate the proportional values of samples
	# all of the above are unbiased estimations of the CDF
	# p = 1. * np.arange(len(data)) / (len(data) - 1)
	# p = (arange(len(x))+0.5)/len(x)
	# p = np.linspace(0, 1, len(data), endpoint=False)
	p = (np.linspace(0, x_len - 1 , x_len)) / x_len
	return x_sort, p


def two_nn_block_analysis(data, frac, num_blocks = 20, shuffle = False):
	"""
	Scale analysis of the two_nn_id algorithm. It takes num_blocks subsamples of data
	and performs the algorithm for each. The size of each subsample is N_samples/i, with i
	going as range(num_blocks).
	:param num_blocks: number of different block sizes to do the scale analysis. \
	If num_blocks = 5, you are scaling down the dataset up to 20% of its original size.
	"""

	# shuffle
	if shuffle:
		np.random.shuffle( data )
		# index = load_data('index.dat')
		# data = data[index.astype(int)]


	N_samples = data.shape[0]
	blocks_dim_avg = []
	blocks_dim_std = []
	blocks_dim_lst = np.arange(1, num_blocks+1)
	# nblock_l = np.linspace(1, num_blocks, num_blocks)



	d_mat2 = dist_mat(data, eucl_dist2)

	for nblock in range(1, num_blocks+1):
		# index = np.arange(N_samples, dtype = int)
		# np.random.shuffle( index )
		rem = N_samples % nblock
		int_div = int(N_samples / nblock)

		dim = []
		for j in range(nblock):
	 		# Divide data points in such that localy: rows = rows_loc ; cols = dim
			size = int_div + 1 - int((nblock - rem + j)/nblock) # it distributes the remainder in a round robin way starting from j = 0. 
			# Not the easiest way of doing it, but I wanted to try without the if().	 		
	 		# Define global row index i
			start = j * (int_div + 1) - int((nblock - rem + j)/nblock) * (j-rem)

			# dim.append(		two_nn_id( r1[	index[int(start):int(start+size)].astype(int)	],
				# r2[	index[int(start):int(start+size)].astype(int)	], frac) 	)

			r1, r2 = calc_two_nn( d_mat2[ int(start):int(start+size),int(start):int(start+size) ] )

			dim_block, _, _ = two_nn_id( r1, r2, frac )
			dim.append(	dim_block )

		blocks_dim_avg.append(np.mean(dim))
		blocks_dim_std.append(np.std(dim))
	return blocks_dim_avg, blocks_dim_std, N_samples/blocks_dim_lst



if __name__ == "__main__":

	data = load_data('./datasets/cauchy20')

	np.random.seed(10)
	
	# dim, mu, mu_cs = two_nn_id(data, 0.9)

	blocks_dim, _, blocks_size = two_nn_block_analysis(data, .9, shuffle = True)

	plt.plot(blocks_size, blocks_dim, "r.-")
	plt.show()