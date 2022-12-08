from pygltflib import GLTF2
import numpy as np
import struct

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


def bounding_box(filename='gltfs/light.gltf', model_mat=None):
    points = load_vertices(filename=filename, model_mat=model_mat)

    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    center = (min_pt + max_pt)/2
    radius = (max_pt - min_pt)/2
    return center, radius

if __name__ == '__main__':
    print(bounding_box())

# (array([-0.00399999, -0.035     ,  0.006     ]), array([0.228, 0.199, 0.228]))
# TODO: write script to merge them, currently merged in blender
