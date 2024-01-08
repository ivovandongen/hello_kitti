#!/usr/bin/env python3
import yaml
import os
import io

DATA_DIR = os.path.dirname(__file__)


def generate_header(names):
    with io.StringIO() as output:
        print("#pragma once\n", file=output)
        print("#include <string>", file=output)
        print("#include <vector>\n", file=output)
        print("namespace ivd::ml::yolo {", file=output)
        print("const std::vector<std::string> class_names{", file=output)
        for index, name in names.items():
            print(f"\t\"{name}\",", file=output)
        print("};", file=output)
        print("} // namespace ivd::ml::yolo", file=output)
        contents = output.getvalue()
    return contents


with open(os.path.join(DATA_DIR, "coco.yaml"), 'r') as f:
    coco = yaml.safe_load(f)
    header_contents = generate_header(coco['names'])
    print(header_contents)

