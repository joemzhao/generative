import numpy as np
import os
import json

actor_dict = json.load(open("actors.json"))
actor_list = actor_dict.values()
filtered_path = "filtered_text.txt"

def remover(file_path):
    out_handle = open(filtered_path, "w")
    with open(file_path, "r") as f:
        while True:
            line = f.readline()
            try:
                _pair = json.loads(line)
            except ValueError:
                pass

            pair = []
            for QA in _pair:
                out = [idx for idx in QA if idx not in actor_list]
                pair.append(out)

            out_handle.write(json.dumps(pair)+"\n")
            pair = []
            if not line:
                break
    f.close()
    out_handle.close()

if __name__ == "__main__":
    file_path = "text.txt"
    remover(file_path)
