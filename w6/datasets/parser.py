import json

file_name = "../datasets/t_given_s_train.txt"
out = "../datasets/train.txt"

out_handle = open(out, "w")

with open(file_name, "r") as f:
    for line in f:
        line = line.strip().split("|")
        Q = map(int, line[0].split())
        A = map(int, line[1].split())
        out_handle.write(json.dumps([Q, A])+"\n")

f.close()
out_handle.close()
