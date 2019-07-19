import numpy as np



def load_data(file_path):
	"""
	load data and return an nd numpy array
	"""
	data = np.loadtxt(file_path, skiprows = 0, unpack = False)
	return data


def load_data_pkl(file_path):
	with open(input_file, 'rb') as file:
        return np.load(file_path)

# def two_nn_id():

def calc_2nn(data):
	for i in data:
		for j in data:

# def compute_id():

# def eucl_dist(x,y):
# 	return x**2 - y**2




if __name__ == "__main__":
	data = load_data('./datasets/cauchy20')
