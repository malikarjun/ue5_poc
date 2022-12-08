from pygltflib import GLTF2
import numpy as np
import struct


# https://ai.stackexchange.com/questions/14041/how-can-i-derive-the-rotation-matrix-from-the-axis-angle-rotation-vector
def rotate_mat(axis_angle):
    k = axis_angle[1:]
    kx, ky, kz = k
    angle = np.deg2rad(axis_angle[0])

    ct = np.cos(angle)
    st = np.sin(angle)

    mat = np.matrix([
        [ct + kx ** 2 * (1 - ct), kx * ky * (1 - ct) - kz * st, kx * kz * (1 - ct) + ky * st, 0.],
        [ky * kx * (1 - ct) + kz * st, ct + ky ** 2 * (1 - ct), ky * kz * (1 - ct) - kx * st, 0.],
        [kz * kx * (1 - ct) - ky * st, kz * ky * (1 - ct) + kx * st, ct + kz ** 2 * (1 - ct), 0.],
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


def load_vertices(filename='gltfs/light.gltf', model_mat=None):
    gltf = GLTF2().load(filename)

    # get the first mesh in the current scene (in this example there is only one scene and one mesh)
    mesh = gltf.meshes[gltf.scenes[gltf.scene].nodes[0]]

    # get the vertices for each primitive in the mesh (in this example there is only one)
    for primitive in mesh.primitives:

        # get the binary data for this mesh primitive from the buffer
        accessor = gltf.accessors[primitive.attributes.POSITION]
        bufferView = gltf.bufferViews[accessor.bufferView]
        buffer = gltf.buffers[bufferView.buffer]
        data = gltf.get_data_from_buffer_uri(buffer.uri)

        # pull each vertex from the binary buffer and convert it into a tuple of python floats
        vertices = []
        for i in range(accessor.count):
            index = bufferView.byteOffset + accessor.byteOffset + i*12  # the location in the buffer of this vertex
            d = data[index:index+12]  # the vertex data
            v = struct.unpack("<fff", d)   # convert from base64 to three floats

            if model_mat is not None:
                vert_homo = np.ones(4)
                vert_homo[:3] = v
                vert_homo = model_mat @ vert_homo
                vert_homo = np.squeeze(np.array(vert_homo))
                v = vert_homo[:3]

            vertices.append(v)

    # convert a numpy array for some manipulation
    vertices = np.array(vertices)
    return vertices


def bounding_box(filename, model_mat=None):
    points = load_vertices(filename=filename, model_mat=model_mat)

    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    center = (min_pt + max_pt)/2
    radius = (max_pt - min_pt)/2
    return center, radius

