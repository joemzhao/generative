import numpy as np
import random

minimum = 1
maximum = 25000
max_len = 15
samples = 10000

with open("neg_train.txt", "w") as f:
    Q_to_write = []
    A_to_write = []
    for i in xrange(samples):
        if i%100 == 0:
            print "Generating %d samples" % i
        for word in xrange(int(max_len*(0.1+random.uniform(0, 1)))):
            Q_to_write.append(random.randint(2, int(maximum*random.uniform(0.1, 1))))
        for word in xrange(int(max_len*(0.1+random.uniform(0, 1)))):
            A_to_write.append(random.randint(2, int(maximum*random.uniform(0.1, 1))))
        for idx, item in enumerate(Q_to_write):
            if (idx+1) == len(Q_to_write):
                f.write("%s" % item)
            else:
                f.write("%s " % item)
        f.write("|")

        for item in A_to_write:
            f.write("%s " % item)

        f.write("\n")
        Q_to_write = []
        A_to_write = []

f.close()
