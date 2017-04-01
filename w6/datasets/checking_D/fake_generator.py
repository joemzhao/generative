import numpy as np
import json

'''
Using the BBT data as fake data for training the LSTM classifier.
concatenate Q and A as one line, and the maximum sentence length is 20
and the largest vocabulary index is 20523
'''

file_name = "filtered_text.txt"
out_file = "fake_D.txt"
out_handle = open(out_file, "w")
max_len = 20
max_idx = 0

with open(file_name, "r") as f:
    for line in f:
        try:
            pair = json.loads(line)
            out_line = pair[0]+pair[1]
            temp = map(int, out_line)

            if len(out_line)<=max_len:
                out_handle.write(json.dumps([out_line, 0])+"\n")
                if max(temp)>max_idx:
                    max_idx = max(temp)

        except Exception:
            pass
    print "The maximum sentence length is: ", max_len
    print "The maximum vocabulary index is: ", max_idx


f.close()
out_handle.close()
