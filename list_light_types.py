from os.path import join
from glob import glob
import re


root_path = "siggraphasia20dataset/code/Routine/scenes/xml"

light_types = {}
for folder in glob(join(root_path, "*")):

	file = join(folder, "main.xml")

	try:
		with open(file, "r") as f:
			content = f.read()
	except Exception:
		continue


	scene_light_types = re.findall("<emitter type=\"[a-z]*\"", content)

	for lt in scene_light_types:
		if lt not in light_types.keys():
			light_types[lt] = 1
		else:
			light_types[lt] += 1


print(light_types)
