import re
import time
import untangle
import os
from os import makedirs
from os.path import basename, dirname, join
import numpy as np
import shutil
import struct
import subprocess
import argparse
from pygltflib import GLTF2, TextureInfo, Texture, Sampler, Node
from pygltflib.utils import ImageFormat, Image


TMP_GLTF_DIR = "gltfs"
TMP_IN_FILE = "tmp_in_file.xml"

AREA_LIGHT_EXTENSION = "KHR_lights_area"
ENV_MAP_EXTENSION = "KHR_lights_environment"

def relative_to_abs_path(content, ref_file):
	pattern = "<string name=\"filename\" value=\"(.*)\""
	res = re.search(pattern, content)
	rel_path = res.group(1)

	if ".." in rel_path:
		ref_folders = ref_file.split("/")
		step_outn = len([_fldr for _fldr in rel_path.split("/") if _fldr == ".."])

		abs_dir_path = "/".join(ref_folders[:-(step_outn+1)])
		rel_dir_path = "/".join([".."] * step_outn)
		content = content.replace(rel_dir_path, abs_dir_path)

	return content


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


def load_vertices(filename, model_mat=None):
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


def load_gltf(filename):
	return GLTF2().load(filename)

def get_scene_nodes(gltf):
	return gltf.scenes[gltf.scene].nodes

def merge_gltf(file1, file2):
	gltf1 = load_gltf(file1)
	gltf2 = load_gltf(file2)

	num_scene_nodes = len(get_scene_nodes(gltf1))
	num_meshes = len(gltf1.meshes)
	num_accessors = len(gltf1.accessors)
	num_buffer_views = len(gltf1.bufferViews)
	num_buffers = len(gltf1.buffers)
	num_materials = len(gltf1.materials)
	num_textures = len(gltf1.textures)
	num_images = len(gltf1.images)
	num_samplers = len(gltf1.samplers)

	# add extensions

	gltf1.extensions = gltf1.extensions | gltf2.extensions
	gltf1.extensionsUsed = list(set().union(*[gltf1.extensionsUsed, gltf2.extensionsUsed]))
	gltf1.extensionsRequired = list(set().union(*[gltf1.extensionsRequired, gltf2.extensionsRequired]))

	# add scene nodes
	for scene_node in get_scene_nodes(gltf2):
		scene_node += num_scene_nodes
		get_scene_nodes(gltf1).append(scene_node)

	# add nodes
	for node in gltf2.nodes:
		if node.mesh is not None:
			node.mesh += num_meshes
		gltf1.nodes.append(node)

	# add meshes
	for mesh in gltf2.meshes:
		for primitive in mesh.primitives:
			primitive.material += num_materials

			primitive.indices += num_accessors
			if primitive.attributes.POSITION is not None:
				primitive.attributes.POSITION += num_accessors
			if primitive.attributes.NORMAL is not None:
				primitive.attributes.NORMAL += num_accessors
			if primitive.attributes.TEXCOORD_0 is not None:
				primitive.attributes.TEXCOORD_0 += num_accessors
			if primitive.attributes.TEXCOORD_1 is not None:
				primitive.attributes.TEXCOORD_1 += num_accessors
		gltf1.meshes.append(mesh)

	# add accessors
	for accessor in gltf2.accessors:
		accessor.bufferView += num_buffer_views
		gltf1.accessors.append(accessor)

	# add buffer views
	for buffer_view in gltf2.bufferViews:
		buffer_view.buffer += num_buffers
		gltf1.bufferViews.append(buffer_view)

	# add buffers
	for buffer in gltf2.buffers:
		gltf1.buffers.append(buffer)

	# add textures
	for texture in gltf2.textures:
		texture.sampler += num_samplers
		texture.source += num_images
		gltf1.textures.append(texture)

	# add samplers
	for sampler in gltf2.samplers:
		gltf1.samplers.append(sampler)

	# add images
	for image in gltf2.images:
		gltf1.images.append(image)

	# add materials
	for material in gltf2.materials:
		if material.normalTexture is not None:
			material.normalTexture.index += num_textures
		if material.pbrMetallicRoughness is not None:
			if material.pbrMetallicRoughness.baseColorTexture is not None:
				material.pbrMetallicRoughness.baseColorTexture.index += num_textures
			if material.pbrMetallicRoughness.metallicRoughnessTexture is not None:
				material.pbrMetallicRoughness.metallicRoughnessTexture.index += num_textures
		gltf1.materials.append(material)

	gltf1.convert_images(ImageFormat.DATAURI)

	return gltf1


def merge_gltfs(files):
	_merged_file = None
	try:
		_merged = merge_gltf(files[0], files[1])
		_merged_file = join(dirname(files[0]), "_merged.gltf")
		_merged.save(_merged_file)

		for file in files[2:]:
			_merged = merge_gltf(_merged_file, file)
			_merged.save(_merged_file)

		gltf = load_gltf(_merged_file)
	finally:
		if _merged_file is not None:
			os.remove(_merged_file)

	return gltf


def get_elem_val(element, atype, aselect_tag, aselect_tag_value, atag=None):
	attrs = getattr(element, atype)

	for attr in attrs:
		if attr[aselect_tag] == aselect_tag_value:
			return attr[atag] if atag is not None else attr

	return None


def get_gltf_material_by_name(gltf, name):
	for material in gltf.materials:
		if name == material.name:
			return material
	return None

def load_textures(gltf, bsdf):
	bsdf_id = bsdf["id"]

	# add sampler
	if len(gltf.samplers) == 0:
		sampler = Sampler()
		sampler.magFilter = 9729
		sampler.minFilter = 9987
		gltf.samplers.append(sampler)

	texture_types = ["albedo", "roughness", "normal"]
	for texture_type in texture_types:
		tex = get_elem_val(bsdf, "texture", "name", texture_type)
		tex_file = get_elem_val(tex, "string", "name", "filename", "value")
		image = Image()
		image.uri = tex_file
		# add image
		gltf.images.append(image)
		gltf.convert_images(ImageFormat.DATAURI)

		# add textures
		texture = Texture()
		texture.sampler = 0
		texture.source = len(gltf.images)-1
		gltf.textures.append(texture)

		# reference the texture by index
		material = get_gltf_material_by_name(gltf, bsdf_id)
		material.doubleSided = True
		if texture_type == "albedo":
			if material.pbrMetallicRoughness.baseColorTexture is None:
				material.pbrMetallicRoughness.baseColorTexture = TextureInfo()
			material.pbrMetallicRoughness.baseColorTexture.index = len(gltf.textures) - 1

		if texture_type == "roughness":
			if material.pbrMetallicRoughness.metallicRoughnessTexture is None:
				material.pbrMetallicRoughness.metallicRoughnessTexture = TextureInfo()
			material.pbrMetallicRoughness.metallicRoughnessTexture.index = len(gltf.textures) - 1

		if texture_type == "normal":
			if material.normalTexture is None:
				material.normalTexture = TextureInfo()
			material.normalTexture.index = len(gltf.textures) - 1

def make_list(val):
	if not isinstance(val, list):
		return [val]
	return val

def parse_xyz(xml):
	attrs = ["x", "y", "z"]
	vals = []
	for attr in attrs:
		vals.append(float(xml[attr]))
	return vals

def parse_axis_angle(xml):
	attrs = ["angle", "x", "y", "z"]
	vals = []
	for attr in attrs:
		vals.append(float(xml[attr]))
	return vals

def handle_scales(scales):
	return [scale_mat(parse_xyz(scale)) for scale in scales]

def handle_rotates(rotates):
	return [rotate_mat(parse_axis_angle(rotate)) for rotate in rotates]


def handle_translates(translates):
	return [translate_mat(parse_xyz(translate)) for translate in translates]

def matrix_mul(mats):
	result = np.eye(4)
	for mat in mats:
		result = mat @ result
	return result

def list_to_mat(lst):
	return np.reshape(np.array(lst), newshape=(4, 4), order="F")

def load_transform(gltf, shape):
	smats = handle_scales(make_list(shape.transform.scale))
	rmats = handle_rotates(make_list(shape.transform.rotate))
	tmats = handle_translates(make_list(shape.transform.translate))

	mmat = np.eye(4)
	midx = min(min(len(smats), len(rmats)), len(tmats))
	for i in range(midx):
		mmat = tmats[i] @ rmats[i] @ smats[i] @ mmat

	if midx < len(smats):
		mmat = matrix_mul(smats[midx:]) @ mmat

	if midx < len(rmats):
		mmat = matrix_mul(rmats[midx:]) @ mmat

	if midx < len(tmats):
		mmat = matrix_mul(tmats[midx:]) @ mmat

	gltf.nodes[0].matrix = column_order(mmat, dtype="float")

def load_area_light(gltf, shape, gltf_file):

	model_mat = list_to_mat(gltf.nodes[0].matrix)
	center, radius = bounding_box(gltf_file, model_mat)

	node = Node()
	node.translation = center.tolist()
	node.scale = radius.tolist()

	node.name = "Area Light"
	node.extensions = {
		AREA_LIGHT_EXTENSION : {
			"light" : 0
		}
	}

	gltf.nodes.append(node)
	gltf.scenes[0].nodes.append(len(gltf.scenes[0].nodes))

	# ensure the shape has emissive material
	gltf.materials[0].emissiveFactor = [3, 3, 3]

	gltf.extensions = {
		AREA_LIGHT_EXTENSION : {
			"lights" : [
				{
					"intensity" : 1.0,
					"color" : [float(val) for val in shape.emitter.rgb["value"].split(" ")],
					"shape" : "rect",
					"width" : 1.0,
					"height" : 1.0
				}
			]
		}
	}

	gltf.extensionsUsed.append(AREA_LIGHT_EXTENSION)
	gltf.extensionsRequired.append(AREA_LIGHT_EXTENSION)



def load_envmap(gltf, envmap, out_dir):
	gltf.extensionsUsed.append(ENV_MAP_EXTENSION)
	gltf.extensionsRequired.append(ENV_MAP_EXTENSION)

	# TODO @mswamy : talk to zhengqin about the scale parameter and how it maps to UE5's intensity
	envmap_file = get_elem_val(envmap, "string", "name", "filename", "value")
	gltf.extensions[ENV_MAP_EXTENSION] = {
		"lights" : [
			{
				"hdri" : basename(envmap_file)
			}
		]
	}
	shutil.copyfile(envmap_file, basename(envmap_file))



def obj2gltf(obj_file, shape, xml, out_dir):
	makedirs(TMP_GLTF_DIR, exist_ok=True)

	out_file = join(TMP_GLTF_DIR, "{}_{}.gltf".format(shape["id"], time.time()))
	cmd = "node main.js {in_file} {out_file}".format(in_file=obj_file, out_file=out_file)
	_ = subprocess.check_output(cmd, shell=True)

	gltf = load_gltf(out_file)
	gltf.extensionsUsed = []
	gltf.extensionsRequired = []
	load_transform(gltf, shape)

	if hasattr(shape, "emitter"):
		load_area_light(gltf, shape, out_file)

	envmap = get_elem_val(xml.scene, "emitter", "type", "envmap")
	if envmap is not None:
		load_envmap(gltf, envmap, out_dir)

	if hasattr(shape, "ref"):
		bsdf_refs = getattr(shape, "ref")
		for bsdf_ref in bsdf_refs:
			bsdf_ref_id = bsdf_ref["id"]

			bsdf = get_elem_val(xml.scene, "bsdf", "id", bsdf_ref_id)
			load_textures(gltf, bsdf)

	# save after textures are loaded
	gltf.save(out_file)
	return out_file


def parse_xml(in_file, out_file):
	gltf_files = []
	try:

		with open(in_file, "r") as fin:
			content = fin.read()
			with open(TMP_IN_FILE, "w") as fout:
				fout.write(relative_to_abs_path(content, in_file))
			in_file = TMP_IN_FILE

		xml = untangle.parse(in_file)
		for shape in xml.scene.shape:
			obj_filename = get_elem_val(shape, "string", "name", "filename", "value")
			# TODO: hack to ignore container.obj, check with zhengqin
			if "container.obj" in obj_filename:
				continue
			gltf_files.append(obj2gltf(obj_filename, shape, xml, dirname(out_file)))

		gltf = merge_gltfs(gltf_files)
	finally:
		for gltf_file in gltf_files:
			os.remove(gltf_file)
		os.remove(TMP_IN_FILE)

	return gltf



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="Input xml file", required=True)
	parser.add_argument("-o", "--output", help="Output gltf like file")

	args = parser.parse_args()
	input_file = args.input
	output_file = "scene.gltf"
	if args.output:
		output_file = args.output

	merged_gltf = parse_xml(input_file, output_file)
	merged_gltf.save(output_file)

