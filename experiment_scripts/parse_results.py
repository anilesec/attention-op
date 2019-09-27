import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from tabulate import tabulate

log_dir = '/tmp-network/user/ppansari/logs/'
def parse_file(filename):
    f = open(filename, 'rU')
    cost_list = []
    time_list = []
    for line in f:
        start = line.find('Average cost')
        start2 = line.find('Average serial duration')
        if start != -1:
            tmp_list = line.split(' ')
            cost_list.append(round(float(tmp_list[2]), 2))
            print(round(float(tmp_list[2]), 2))
        if start2 != -1:
            tmp_list = line.split(' ')
            time_list.append(round(float(tmp_list[3]), 2))
            print(round(float(tmp_list[3]), 2))

    return cost_list, time_list
    f.close()

if __name__ == '__main__':
    c1, t1 = parse_file(log_dir + sys.argv[1])
    c2, t2 = parse_file(log_dir + sys.argv[2])
    c3, t3 = parse_file(log_dir + sys.argv[3])
    c4, t4 = parse_file(log_dir + sys.argv[4])
    c2 = [x/y for x, y in zip(c2, c1)]
    c3 = [x/y for x, y in zip(c3, c1)]
    c4 = [x/y for x, y in zip(c4, c1)]
    c1 = [x/y for x, y in zip(c1, c1)]
#    x_ax = range(100, 1001, 100)
#    plt.plot(x_ax, c1, label = 'Concorde')
#    plt.plot(x_ax, c2, label = 'Farthest Insertion')
#    plt.plot(x_ax, c3, label = 'AM-20')
#    plt.plot(x_ax, c4, label = 'AM-100')
#    plt.savefig('temp.png')
    c1.insert(0, 'Concorde')
    c2.insert(0, 'Farthest Insertion')
    c3.insert(0, 'AM-20')
    c4.insert(0, 'AM-100')
    head = ['Method']
    f = open('cost-table.txt', 'w')
    for t in range(100, 1001, 100):
        head.append('n = ' + str(t))
    f.write(tabulate([c1, c2, c3, c4], headers = head, tablefmt='orgtbl'))
    f.close()
    print(tabulate([c1, c2, c3, c4], headers = head, tablefmt='orgtbl'))

    t1.insert(0, 'Concorde')
    t2.insert(0, 'Farthest Insertion')
    t3.insert(0, 'AM-20')
    t4.insert(0, 'AM-100')
    head = ['Method']
    f = open('time-table.txt', 'w')
    for t in range(100, 1001, 100):
        head.append('n = ' + str(t))
    f.write(tabulate([t1, t2, t3, t4], headers = head, tablefmt='orgtbl'))
    f.close()
    print(tabulate([t1, t2, t3, t4], headers = head, tablefmt='orgtbl'))


