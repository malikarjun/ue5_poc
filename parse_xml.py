import time

import untangle
import os
from os import makedirs
from os.path import basename, dirname, join
from gltf_utils import load_gltf, merge_gltfs
import numpy as np
import shutil

import subprocess
import argparse

from pygltflib import TextureInfo, Texture, Sampler, Node
from pygltflib.utils import ImageFormat, Image

from utils import scale_mat, rotate_mat, translate_mat, column_order
from utils import bounding_box



TMP_GLTF_DIR = "gltfs"
AREA_LIGHT_EXTENSION = "KHR_lights_area"
ENV_MAP_EXTENSION = "KHR_lights_environment"


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
	# mat = np.eye(4)
	# for scale in scales:
	# 	mat = scale_mat(parse_xyz(scale)) @ mat
	# return mat

	return [scale_mat(parse_xyz(scale)) for scale in scales]

def handle_rotates(rotates):
	# mat = np.eye(4)
	# for rotate in rotates:
	# 	mat = rotate_mat(parse_axis_angle(rotate)) @ mat
	# return mat
	return [rotate_mat(parse_axis_angle(rotate)) for rotate in rotates]


def handle_translates(translates):
	# mat = np.eye(4)
	# for translate in translates:
	# 	mat = translate_mat(parse_xyz(translate)) @ mat
	# return mat
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
	cmd = "obj2gltf -i {in_file} -o {out_file}".format(in_file=obj_file, out_file=out_file)
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


	# merged_gltf = parse_xml("siggraphasia20dataset/code/Routine/scenes/xml/scene0065_01/main.xml")
	# merged_gltf.save("merged_final.gltf")

