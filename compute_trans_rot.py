import re

import numpy as np

translation = np.zeros(3)
with open('translation.txt', 'r') as f:
	for line in f.readlines():
		translation += np.array([float(val) for val in re.findall("-?\d+.\d*", line)])

print('{}, {}, {}'.format(*translation))


def axis_angle_to_quaternions(angle, x, y, z):
	angle *= np.pi/180
	qx = x * np.sin(angle / 2)
	qy = y * np.sin(angle / 2)
	qz = z * np.sin(angle / 2)
	qw = np.cos(angle / 2)
	return [qx, qy, qz, qw]

def read_rotation_file():
	with open('rotation.txt', 'r') as f:
		return [float(val) for val in re.findall("-?\d+.\d*", f.readline())]

print('{}, {}, {}, {}'.format(*axis_angle_to_quaternions(*read_rotation_file())))