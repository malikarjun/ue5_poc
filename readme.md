# xml2gltf

This script can be used to convert xml files of [OptixRenderer](https://github.com/lzqsd/OptixRenderer/) to 
gltf like files which can be imported into UE5.

## Prerequisites
1. Install dependencies

`pip install -r requirements.txt`

2. Install nodejs

```
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.2/install.sh | bash
chmod +x ~/.nvm/nvm.sh
source ~/.bashrc 

nvm install 16
```

## Run script

`python script.py -i {abs path to xml file}`

Two files are generated as output in the current working directory. 
1. scene.gltf
2. *.hdr file (this is only generated if the scene contains envmap)

Copy both these files and store them in the same directory which can be accessed by Unreal Engine.