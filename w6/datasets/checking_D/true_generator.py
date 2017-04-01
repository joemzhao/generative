import json

'''
Parse the dataset to generate real training data for the LSTM classifier.
concatenate Q and A and the maximum length is 20, largest vocab index is
24991
'''

file_name = "pre_true_D.txt"
out_file = "true_D.txt"
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
                out_handle.write(json.dumps([out_line, 1])+"\n")
                if max(temp)>max_idx:
                    max_idx = max(temp)

        except Exception:
            pass

    print "The maximum sentence length is: ", max_len
    print "The maximum vocabulary index is: ", max_idx

f.close()
out_handle.close()
