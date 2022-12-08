const obj2gltf = require("obj2gltf");
const fs = require("fs");

let input = process.argv[2]
let output = process.argv[3]


obj2gltf(input).then(function (gltf) {
  const data = Buffer.from(JSON.stringify(gltf));
  fs.writeFileSync(output, data);
});