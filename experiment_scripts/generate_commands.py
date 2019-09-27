#!/usr/bin/env python

import sys
import math
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import subprocess

def main():
#    graph_sizes = [10, 20]
    graph_sizes = range(100, 1100, 100) 
    fp = open('command_list.sh', 'w')
    for t in graph_sizes:
        cmd = "python generate_data.py --problem tsp --name test --seed 2345 --dataset_size 100 --graph_sizes " + str(int(t)) + " -f"
        fp.write(cmd + '\n')
        print(cmd)
    fp.close()
if __name__ == '__main__':
    main()
