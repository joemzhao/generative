import numpy as np
import json

def spliter(ls_path):
    whole = []
    with open(ls_path, "r") as ls:
        for line in ls:
            line = line.strip().split()
            parse_line = [int(x) for x in line]
            whole.append(parse_line)
    ls.close()
    return whole



def mapping(json_path, val_list):
    _j = open(json_path).read()
    _dict = json.loads(_j)

    ret = []
    for val in val_list:
        for idx in val:
            for key, value in _dict.iteritems():
                if idx == value:
                    ret.append(key)
    return ret

if __name__ == "__main__":
    js_path = "text_data/dict.json"
    ls_path = "target_generate/eval_file_of_pretrain.txt"
    full = spliter(ls_path)
    print mapping(js_path, full)[:20]
