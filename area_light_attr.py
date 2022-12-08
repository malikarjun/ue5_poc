import numpy as np
import re
from load_vertices import *

KINDA_SMALL_NUMBER = 1e-4


# https://ai.stackexchange.com/questions/14041/how-can-i-derive-the-rotation-matrix-from-the-axis-angle-rotation-vector
def rotate_mat(axis_angle):
	k = axis_angle[1:]
	kx, ky, kz = k
	angle = np.deg2rad(axis_angle[0])

	ct = np.cos(angle)
	st = np.sin(angle)
	
	mat = np.matrix([
		[ct + kx ** 2 * (1 - ct)     ,    kx * ky * (1 - ct) - kz * st,    kx * kz * (1 - ct) + ky * st,   0.],
		[ky * kx * (1 - ct) + kz * st,    ct + ky ** 2 * (1 - ct)     ,    ky * kz * (1 - ct) - kx * st,   0.],
		[kz * kx * (1 - ct) - ky * st,    kz * ky * (1 - ct) + kx * st,    ct + kz ** 2 * (1 - ct),        0.],
		[0., 0., 0., 1.]
	])

	return mat

def translate_mat(translate):
	mat = np.eye(4)
	for i in range(3):
		mat[i, 3] = translate[i]

	return mat

def scale_mat(scale):
	mat = np.eye(4)
	for i in range(3):
		mat[i, i] = scale[i]
	return mat

def parse(content):
	return np.array([float(val) for val in re.findall("-?\d+.\d*", content)])

def array_to_string(data):
	str_val = ""
	for val in data:
		str_val += "{:.5f}, ".format(val)

	str_val = str_val[:-2]
	return str_val


def column_order(matrix, dtype="str"):
	data = []
	r, c = matrix.shape
	for j in range(c):
		for i in range(r):
			if dtype == "str":
				data.append("{:.5f}".format(matrix[i, j]))
			else:
				data.append(matrix[i, j])
	return data


def compute_model_mat(filename='transformation.txt'):
	matrix = np.eye(4)
	with open(filename, 'r') as f:
		for line in f.readlines():
			if 'scale' in line:
				scale = scale_mat(parse(line))
				# print(scale)
				matrix = scale @ matrix
				# matrix = matrix @ scale
			elif 'rotate' in line:
				rotate = rotate_mat(parse(line))
				# print(rotate)
				matrix = rotate @ matrix
				# matrix = matrix @ rotate
			elif 'translate' in line:
				translate = translate_mat(parse(line))
				# print(translate)
				matrix = translate @ matrix
				# matrix = matrix @ translate

	# print(repr(matrix.flatten('F')))
	return column_order(matrix), matrix

def normalize(vec):
	return vec/np.linalg.norm(vec)

def make_from_yx(y_axis, x_axis):
	new_y = normalize(y_axis)
	norm = normalize(x_axis)

	# if they're almost same, we need to find arbitrary vector
	if (1 - np.abs(np.dot(new_y, norm))) < 0.00001:
		norm = np.array([0, 0, 1]) if (np.abs(new_y[2]) < (1. - KINDA_SMALL_NUMBER)) else np.array([1, 0, 0])

	new_z = normalize(np.cross(norm, new_y))
	new_x = np.cross(new_y, new_z)

	matrix = np.eye(4)
	matrix[0, :3] = new_x
	matrix[1, :3] = new_y
	matrix[2, :3] = new_z

	return matrix

def make_from_xz(x_axis, z_axis):
	new_x = normalize(x_axis)
	norm = normalize(z_axis)

	# if they're almost same, we need to find arbitrary vector
	if (1 - np.abs(np.dot(new_x, norm))) < 0.00001:
		norm = np.array([0, 0, 1]) if (np.abs(new_x[2]) < (1. - KINDA_SMALL_NUMBER)) else np.array([1, 0, 0])

	new_y = normalize(np.cross(norm, new_x))
	new_z = np.cross(new_x, new_y)

	matrix = np.eye(4)
	matrix[0, :3] = new_x
	matrix[1, :3] = new_y
	matrix[2, :3] = new_z

	return matrix


def matrix_to_quat(matrix):
	tr = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]

	if tr > 0.0:
		inv_s = 1/np.sqrt(tr + 1.)
		w = 0.5 * (1. / inv_s)
		s = 0.5 * inv_s

		x = (matrix[1, 2] - matrix[2, 1]) * s
		y = (matrix[2, 0] - matrix[0, 2]) * s
		z = (matrix[0, 1] - matrix[1, 0]) * s
	else:
		# diagonal is negative
		i = 0
		if matrix[1, 1] > matrix[0, 0]:
			i = 1

		if matrix[2, 2] > matrix[i, i]:
			i = 2

		nxt = [1, 2, 0]
		j = nxt[i]
		k = nxt[j]

		s = matrix[i, i] - matrix[j, j] - matrix[k, k] + 1.0

		inv_s = 1/np.sqrt(s)

		qt = np.zeros(4)
		qt[i] = 0.5 * (1./inv_s)

		s = 0.5 * inv_s

		qt[3] = (matrix[j, k] - matrix[k, j]) * s
		qt[j] = (matrix[i, j] - matrix[j, i]) * s
		qt[k] = (matrix[i, k] - matrix[k, i]) * s

		x, y, z, w = qt
	return np.array([w, x, y, z])


def compute_trans_and_rot(center, direction, scale=1.0):

	forward = direction
	# TODO: this is same as unreal, maybe we need a different one here
	world_up = np.array([0, 0, 1])

	# rot_matrix = make_from_xz(x_axis=forward, z_axis=world_up)
	rot_matrix = make_from_yx(y_axis=forward, x_axis=world_up)

	# TODO: this minus sign was added to match the manual results, this is extremely hacky!!
	rot_quat = -matrix_to_quat(rot_matrix)
	trans = center + scale * direction

	return trans, rot_quat


def _compute_trans_and_rot(center, direction):
	direction = normalize(direction)


if __name__ == '__main__':
	data, model_mat = compute_model_mat()
	center, radius = bounding_box(model_mat=model_mat)
	print(center, radius)
	# print(center - np.array([0, radius[1], 0]))

	# direction = np.array([0, radius[1], 0])
	# print(compute_trans_and_rot(center=center, direction=direction))
	# print(compute_trans_and_rot(center=center, direction=direction, scale=5))

	# direction = -np.array([0, 0, radius[2]])
	# print(compute_trans_and_rot(center=center, direction=direction))
	# print(compute_trans_and_rot(center=center, direction=direction, scale=5))



	# print(', '.join(data))