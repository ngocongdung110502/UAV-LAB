"""
    Plot result
"""


import json
import numpy as np
import matplotlib.pyplot as plt


def read_file(filename):
    f = open(filename, "r")
    res = []
    line = f.readline().strip()
    while line != "":
        x = json.loads(line)
        res.append(x)
        line = f.readline().strip()
    x = np.asarray(res)
    p = x.mean(axis=0)
    return p.tolist()


x1 = read_file("./change/choose/dautranh.txt")
x2 = read_file("./change/choose/khongphancap.txt")
x3 = read_file("./change/choose/phancap.txt")
# x4 = read_file("./change/size/size150.txt")
# x5 = read_file("./change/size/size200.txt")
# x4 = read_file("size_100.txt")
# x5 = read_file("size_150.txt")
# x6 = read_file("size_200.txt")

y = [i*100 for i in range(len(x1))]

plt.plot(y, x1, label="Đấu tranh")
plt.plot(y, x2, color="green", label="Không phân cấp")
plt.plot(y, x3, color="red", label="Phân cấp")
# plt.plot(y, x4, color="blue", label="150")
# plt.plot(y, x5, color="orange", label="200")
plt.xlabel("Số thế hệ")
plt.ylabel("Hàm mục tiêu")
plt.legend()
plt.show()
