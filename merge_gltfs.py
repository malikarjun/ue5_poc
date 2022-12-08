import os
import shutil

import gltflib
from pygltflib import GLTF2, BufferFormat, ImageFormat
from pygltflib.utils import glb2gltf, gltf2glb
from os.path import dirname, join

import numpy as np
import struct


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
			# primitive.material = None

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


if __name__ == "__main__":
	_merged = merge_gltf("gltfs/bathtub.gltf", "gltfs/toilet_seat.gltf")
	_merged.save("gltfs/merged.gltf")
	_merged = merge_gltf("gltfs/merged.gltf", "gltfs/wash_basin.gltf")
	_merged.save("gltfs/merged.gltf")
	_merged = merge_gltf("gltfs/merged.gltf", "gltfs/room.gltf")
	_merged.save("gltfs/merged.gltf")

