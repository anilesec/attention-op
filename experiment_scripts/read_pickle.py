import pickle
import numpy as np

objects = []

with(open("./results/tsp/tsp100_test_seed1234/tsp100_test_seed1234-pretrained_tsp_100-greedy-t1-0-10000.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break


#print(isinstance(objects, list))
#print(type(objects))
#print(objects[0])
temp = []
for x in objects[0][0]:
    temp.append(x[0])
#    print(x[0])

print(np.mean(np.array(temp)))
#for t in len(temp):
#    print(t)
#for t in range(len(objects)):
#    if t < 5:
#        print(objects[t])
