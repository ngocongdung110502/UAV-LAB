import json
from math import sqrt, log2
import math
import random
import matplotlib.pyplot as plt
import numpy as np

H = 10
V = 20
T = 20
t = 0.5
d = V*t
X = 20
Y = 10
Y2 = 5
w_s = [5, 0]
w_d = [15, 0]
q_I1 = [0, 10]
q_F1 = [20, 10]
q_I2 = [0, 5]
q_F2 = [20, 5]
N = int(T/t)
N1 = N + 1

ro = 0.012
s = 0.05
A = 0.8
om = 100
R = 0.08
I = 0.1
p = 1.225
W = 0.5
# v_0 = 7.2
d_0 = 0.0151/s/A
e = math.e
E = 0.5772156649
alpha = 2.2
a2 = alpha/2
B = 20
w_0 = 10**(-3)

xichma = 0.25
n_u = 0.5
P_u = 5  # 7.5 mW
P_b = 10**(-3)
P_s = 10**(1.6)  # 16dBm
xich_ma_u = 10**(-6)
nguy = 0.5
P_wpt = 1*10**7
S = 50
micro = 0.84

v_0 = sqrt(W / (2 * p * A))

P_0 = ro / 8 * p * s * A * om**3 * R ** 3
k_1 = 3 / om**2 / R**2
P_1 = (1 + I) * W**1.5 / sqrt(2*p*A)
k_2 = 1 / 2 / v_0**2
k_3 = 0.5 * d_0 * p * s * A


def ceil(x: float) -> int:
    """
        Calculate ceil of float number\n
        @parameter:
            x: float | int
        @return: ceil of x
        Example:
            ceil(0.7) = 1
            ceil(-1.5) = 1

    """
    _x = int(x)
    if _x == x:
        return _x
    else:
        return _x + 1


def distance2(x, y):
    """
        Calculate distance ^2 between two point X, Y
            X(x1, y1), Y(x2, y2)
        return distance^2
    """
    return (x[0]-y[0])**2 + (x[1]-y[1])**2


def distance(x, y):
    """
        Calculate distance euclid between two point X, Y
            X(x1, y1), Y(x2, y2)
        return distance
    """
    return sqrt(distance2(x, y))


P_u_bar = P_u * (1 + ceil(xichma))
theta = e**(-E) * w_0 / xich_ma_u


def fitness1(c, log=False):
    """
        fitness function of individial
        @parameter:
            c: individial : list
        @return
            * 0 : if fail constraint: distance, E_fly, total rate
            * rate/2 : if only fail rate
            * rate : if pass all constraint
    """
    x = c[:N1]  # x location
    y = c[N1:N1+N1]  # y location
    z = c[N1+N1:]  # tau
    # convert (0, 1) to real
    l = [[x[i]*X, y[i]*Y] for i in range(N1)]  # location
    e_fly1 = 0  # total fly energy
    e_h1 = 0  # total harvested energy
    r_u = 0  # total rate from source -> uav
    r_d = 0  # total rate from uav -> destination
    for i in range(0, N):
        d_su2 = distance2(l[i], w_s)  # distance^2 from uav to source
        d_du2 = distance2(l[i], w_d)  # distance^2 from uav to destination

        # R_u at time slot i
        # formula (20)
        r_u_i = log2(1 + (theta * P_s) / ((H**2 + d_su2)**a2))
        r_u = r_u + 1 * z[i] * t * r_u_i

        _o = ((H**2 + d_su2)**a2 * (H**2 + d_du2)**a2)
        # formula (21)
        r_d_i1 = log2(1+(theta*(n_u*w_0*P_s+P_u_bar*(H**2 + d_su2)**a2)) / _o)

        # Data transmission rate from UAV to d if it cached a part of f file
        r_d_i2 = log2(1 + theta * P_u / (H**2 + d_du2)**a2)
        # R_d at time slot i
        r_d_i = r_d_i1
        # r_d_i = r_d_i1 + r_d_i2

        r_d = r_d + 1 * z[i] * t * r_d_i
    for i in range(0, N1-1):
        q1, q0 = l[i+1], l[i]
        dis = distance(q1, q0)

        # e_fly at time slot i
        # formula (8)
        e_fly_i = P_0 * (t + k_1 * dis**2) + P_1 * sqrt(sqrt(t**4 + k_2**2 * dis**4) -
                                                        k_2 * dis**2) + k_3 * dis**3 / t**2 + z[i] * t * (P_b + P_u)
        e_fly1 = e_fly1 + e_fly_i

        # e_h at time slot i
        # formula (10)
        e_h_i = micro*(1 - z[i])*t*w_0*P_wpt/(H**2 + distance(q0, w_s)**2)**a2
        e_h1 = e_h1 + e_h_i

    if log:
        print(
            f"E_fly1= {e_fly1}, E_H1= {e_h1}, R_d1= {r_d}, R_u1 + xichma*S = {r_u + xichma * S}")
    # constraint 22e
    for i in range(0, N1-1):
        if distance(l[i], l[i+1]) > d:
            return 0
    # constraint 22d
    if e_fly1 > e_h1:
        return 0
    # constraint 22b
    if r_u + xichma * S < r_d:
        return 0
    # constraint 22c
    # soften constraint
    if r_d < S:
        return r_d / 2
    # pass all constraint
    return r_d

def fitness2(c, log=False):
    """
        fitness function of individial
        @parameter:
            c: individial : list
        @return
            * 0 : if fail constraint: distance, E_fly, total rate
            * rate/2 : if only fail rate
            * rate : if pass all constraint
    """
    x = c[:N1]  # x location
    y = c[N1:N1+N1]  # y location
    z = c[N1+N1:]  # tau
    # convert (0, 1) to real
    l = [[x[i]*X, y[i]*Y2] for i in range(N1)]  # location
    e_fly = 0  # total fly energy
    e_h = 0  # total harvested energy
    r_u = 0  # total rate from source -> uav
    r_d = 0  # total rate from uav -> destination
    for i in range(0, N):
        d_su2 = distance2(l[i], w_s)  # distance^2 from uav to source
        d_du2 = distance2(l[i], w_d)  # distance^2 from uav to destination

        # R_u at time slot i
        # formula (20)
        r_u_i = log2(1 + (theta * P_s) / ((H**2 + d_su2)**a2))
        r_u = r_u + 1 * z[i] * t * r_u_i

        _o = ((H**2 + d_su2)**a2 * (H**2 + d_du2)**a2)
        # formula (21)
        r_d_i1 = log2(1+(theta*(n_u*w_0*P_s+P_u_bar*(H**2 + d_su2)**a2)) / _o)

        # Data transmission rate from UAV to d if it cached a part of f file
        r_d_i2 = log2(1 + theta * P_u / (H**2 + d_du2)**a2)
        # R_d at time slot i
        r_d_i = r_d_i1
        # r_d_i = r_d_i1 + r_d_i2

        r_d = r_d + 1 * z[i] * t * r_d_i
    for i in range(0, N1-1):
        q1, q0 = l[i+1], l[i]
        dis = distance(q1, q0)

        # e_fly at time slot i
        # formula (8)
        e_fly_i = P_0 * (t + k_1 * dis**2) + P_1 * sqrt(sqrt(t**4 + k_2**2 * dis**4) -
                                                        k_2 * dis**2) + k_3 * dis**3 / t**2 + z[i] * t * (P_b + P_u)
        e_fly = e_fly + e_fly_i

        # e_h at time slot i
        # formula (10)
        e_h_i = micro*(1 - z[i])*t*w_0*P_wpt/(H**2 + distance(q0, w_s)**2)**a2
        e_h = e_h + e_h_i

    if log:
        print(
            f"E_fly2= {e_fly}, E_H2= {e_h}, R_d2= {r_d}, R_u2 + xichma*S = {r_u + xichma * S}")
    # constraint 22e
    for i in range(0, N1-1):
        if distance(l[i], l[i+1]) > d:
            return 0
    # constraint 22d
    if e_fly > e_h:
        return 0
    # constraint 22b
    if r_u + xichma * S < r_d:
        return 0
    # constraint 22c
    # soften constraint
    if r_d < S:
        return r_d / 2
    # pass all constraint
    return r_d

def init_one_child1():
    """
        init one individial
        @return:
            * individial: if pass fitness function
            * None: if not pass fitness function
    """
    z = [random.uniform(0, 1)]  # tau
    x = [0]  # x location
    y = [1]  # y location
    e_fly = 0  # total fly energy
    e_h = 0  # total harvested energy
    for _ in range(1, N1-1):
        i, j = x[-1], y[-1]  # x, y location at time slot i - 1
        i, j = i * X, j * Y  # scale from (0, 1) to real
        _i = random.uniform(0, 1)  # random x location at time slot i
        _j = random.uniform(0, 1)  # random y location at time slot i
        _z = random.uniform(0, 1)  # random tau at time slot i
        dis = distance([i, j], [_i * X, _j * Y])  # distance from Q_i-1 to Q_i
        e_fly_i = P_0 * (t + k_1 * dis**2) + P_1 * sqrt(sqrt(t**4 + k_2**2 * dis**4) -
                                                        k_2 * dis**2) + k_3 * dis**3 / t**2 + _z * t * (P_b + P_u)
        e_h_i = micro * (1 - _z) * t * w_0 * \
            P_wpt / (H**2 + distance([i, j], w_s))**a2
        k = 0  # checkpoint when not found next satisfied point
        while dis > d or (e_fly + e_fly_i) > (e_h + e_h_i):
            k += 1
            if k == 100:
                return None
            _i = random.uniform(0, 1)
            _j = random.uniform(0, 1)
            _z = random.uniform(0, 1)
            dis = distance([i, j], [_i * X, _j * Y])
            e_fly_i = P_0 * (t + k_1 * dis**2) + P_1 * sqrt(sqrt(t**4 + k_2**2 * dis**4) -
                                                            k_2 * dis**2) + k_3 * dis**3 / t**2 + _z * t * (P_b + P_u)
            e_h_i = micro * (1 - _z) * t * w_0 * \
                P_wpt / (H**2 + distance([i, j], w_s))**a2
        x.append(_i)
        y.append(_j)
        z.append(_z)
        e_fly = e_fly + e_fly_i
        e_h = e_h + e_h_i
    x.append(1)
    y.append(1)
    z.append(random.uniform(0, 1))
    v = fitness1(x + y + z)
    if v > 0:
        return x + y + z
    else:
        return None
    
def init_one_child2():
    """
        init one individial
        @return:
            * individial: if pass fitness function
            * None: if not pass fitness function
    """
    z = [random.uniform(0, 1)]  # tau
    x = [0]  # x location
    y = [1]  # y location
    e_fly = 0  # total fly energy
    e_h = 0  # total harvested energy
    for _ in range(1, N1-1):
        i, j = x[-1], y[-1]  # x, y location at time slot i - 1
        i, j = i * X, j * Y2  # scale from (0, 1) to real
        _i = random.uniform(0, 1)  # random x location at time slot i
        _j = random.uniform(0, 1)  # random y location at time slot i
        _z = random.uniform(0, 1)  # random tau at time slot i
        dis = distance([i, j], [_i * X, _j * Y2])  # distance from Q_i-1 to Q_i
        e_fly_i = P_0 * (t + k_1 * dis**2) + P_1 * sqrt(sqrt(t**4 + k_2**2 * dis**4) -
                                                        k_2 * dis**2) + k_3 * dis**3 / t**2 + _z * t * (P_b + P_u)
        e_h_i = micro * (1 - _z) * t * w_0 * \
            P_wpt / (H**2 + distance([i, j], w_s))**a2
        k = 0  # checkpoint when not found next satisfied point
        while dis > d or (e_fly + e_fly_i) > (e_h + e_h_i):
            k += 1
            if k == 100:
                return None
            _i = random.uniform(0, 1)
            _j = random.uniform(0, 1)
            _z = random.uniform(0, 1)
            dis = distance([i, j], [_i * X, _j * Y2])
            e_fly_i = P_0 * (t + k_1 * dis**2) + P_1 * sqrt(sqrt(t**4 + k_2**2 * dis**4) -
                                                            k_2 * dis**2) + k_3 * dis**3 / t**2 + _z * t * (P_b + P_u)
            e_h_i = micro * (1 - _z) * t * w_0 * \
                P_wpt / (H**2 + distance([i, j], w_s))**a2
        x.append(_i)
        y.append(_j)
        z.append(_z)
        e_fly = e_fly + e_fly_i
        e_h = e_h + e_h_i
    x.append(1)
    y.append(1)
    z.append(random.uniform(0, 1))
    v = fitness2(x + y + z)
    if v > 0:
        return x + y + z
    else:
        return None


def linear_cross_over(_f, _m):
    """
        Cross over from parents
        @parameter:
            _f: father
            _m: mother
        @return: 2 children from parents
    """
    c1, c2 = _f.copy(), _m.copy()
    u = random.uniform(0, 1)
    for i in range(3*N1):
        c1[i] = u * _f[i] + (1 - u)*_m[i]
        c2[i] = (1 - u)*_f[i] + u * _m[i]
    return c1, c2



def laplace_cross_over(_f, _m):
    c1, c2 = _f.copy(), _m.copy()
    b = np.random.laplace(0., 1., 1)[0]/10
    for i in range(3*N1):
        c1[i] = _f[i] + b * abs(_f[i] - _m[i])
        c2[i] = _m[i] + b * abs(_f[i] - _m[i])
    return c1, c2


X1 = list(range(1, N1-1))  # [1 -> 39]
X2 = list(range(N+2, 2*N+1))  # [1-39]
X3 = list(range(2*N+2, 3*N + 3))  # [0-40]


def random_mutation(c):
    """
        mutaion function
        @parameter:
            c: individual
    """
    x1 = random.choice(X1)
    x2 = random.choice(X2)
    x3 = random.choice(X3)
    p = random.uniform(-1, 1)
    if p < 0:
        p = 0
    c[x1] = random.uniform(0, 1)
    c[x2] = p
    c[x3] = random.uniform(0, 1)


def choose1(population):
    """
        Random 2 children of population a, b
        choose a if fitness of a better than b

        @parameter:
            population: population
        @return
            new polulation of size
    """
    x = population.copy()
    random.shuffle(x)
    res = []
    for i in range(len(x)//2):
        a, b = x[i], x[len(x)//2 + i]
        if fitness1(a) >= fitness1(b):
            res.append(a)
        else:
            res.append(b)
    return res

def choose2(population):
    """
        Random 2 children of population a, b
        choose a if fitness of a better than b

        @parameter:
            population: population
        @return
            new polulation of size
    """
    x = population.copy()
    random.shuffle(x)
    res = []
    for i in range(len(x)//2):
        a, b = x[i], x[len(x)//2 + i]
        if fitness2(a) >= fitness2(b):
            res.append(a)
        else:
            res.append(b)
    return res


def init_population1(size):
    """
        init population
        @parameter:
            size: number of population
        @return: population
    """
    a = []
    for i in range(size):
        o = init_one_child1()
        while o is None:
            print("Init population ...", i)
            o = init_one_child1()
        a.append(o)
    return a

def init_population2(size):
    """
        init population
        @parameter:
            size: number of population
        @return: population
    """
    a = []
    for i in range(size):
        o = init_one_child2()
        while o is None:
            print("Init population ...", i)
            o = init_one_child2()
        a.append(o)
    return a

size = 100
nums_generation = 10000
mutation_rate = 0.1

population1 = init_population1(size)
population2 = init_population2(size)

plt.ion()
figure, ax = plt.subplots()
x1 = np.array(population1[0][:N1]) * X
y1 = np.array(population1[0][N1:N1+N1]) * Y
line1,  = ax.plot(x1, y1, marker="o", linewidth=2, markersize=6, label = 'UAV_1')

x2 = np.array(population2[0][:N1]) * X
y2 = np.array(population2[0][N1:N1+N1]) * Y2
line2,  = ax.plot(x2, y2, marker="o", linewidth=2, markersize=6, label = 'UAV_2')

ax.legend()

ax.scatter(w_s[0], w_s[1], color="red", s=50)
ax.text(w_s[0], w_s[1], "Source", color="blue")
ax.scatter(w_d[0], w_d[1], color="blue", s=50)
ax.text(w_d[0], w_d[1], "Destination", color="red")

ax.scatter(q_I1[0], q_I1[1], color="green", s=50)
ax.text(q_I1[0], q_I1[1], "Start1", color="green")
ax.scatter(q_F1[0], q_F1[1], color="yellow", s=50)
ax.text(q_F1[0], q_F1[1], "Finish1", color="green")

ax.scatter(q_I2[0], q_I2[1], color="green", s=50)
ax.text(q_I2[0], q_I2[1], "Start2", color="green")
ax.scatter(q_F2[0], q_F2[1], color="yellow", s=50)
ax.text(q_F2[0], q_F2[1], "Finish2", color="green")

g = []

population1.sort(key=fitness1, reverse=True)
population2.sort(key=fitness2, reverse=True)
for i in range(nums_generation):
    # print(f"{i}")
    best = fitness1(population1[0]) + fitness2(population2[0])
    print(f"{i} Best: {best}")
    if i % 100 == 0:
        if best < S:
            g.append(best*2)
        else:
            g.append(best)

    x1 = np.array(population1[0][:N1]) * X
    y1 = np.array(population1[0][N1:N1+N1]) * Y
    x2 = np.array(population2[0][:N1]) * X
    y2 = np.array(population2[0][N1:N1+N1]) * Y2
    line1.set_xdata(x1)
    line1.set_ydata(y1)
    line2.set_xdata(x2)
    line2.set_ydata(y2)

    figure.canvas.draw()
    figure.canvas.flush_events()

    # start cross over
    next1 = []
    for i in range(size//2):
        father, mother = random.sample(population1, 2)
        x, y = linear_cross_over(father, mother)
        next1.append(x)
        next1.append(y)

    # mutation
    muta = random.sample(next1, int(mutation_rate*size))
    for i in muta:
        random_mutation(i)

    # group population
    population1 = population1 + next1

    # sort by fitness function
    # population = choose(population)
    population1.sort(key=fitness1, reverse=True)

    # population = population[:size] # choose 50% best

    # choose 40% best, 30% medium, 30% bad
    _size = 2 * size
    t2 = int(size * 2 / 5)
    t3 = int(size / 5)
    t1 = size - t2 - t3
    population1 = population1[:t1] + \
        population1[_size//3:_size//3+t2] + population1[-t3:]
    
    next2 = []
    for i in range(size//2):
        father, mother = random.sample(population2, 2)
        x, y = linear_cross_over(father, mother)
        next2.append(x)
        next2.append(y)

    # mutation
    muta = random.sample(next2, int(mutation_rate*size))
    for i in muta:
        random_mutation(i)

    # group population
    population2 = population2 + next2

    # sort by fitness function
    # population = choose(population)
    population2.sort(key=fitness2, reverse=True)

    # population = population[:size] # choose 50% best

    # choose 40% best, 30% medium, 30% bad
    _size = 2 * size
    t2 = int(size * 2 / 5)
    t3 = int(size / 5)
    t1 = size - t2 - t3
    population2 = population2[:t1] + \
        population2[_size//3:_size//3+t2] + population2[-t3:]



# print result
population1.sort(key=fitness1, reverse=True)
population2.sort(key=fitness2, reverse=True)
plt.ioff()
plt.close()
x1 = np.array(population1[0][:N1]) * X
y1 = np.array(population1[0][N1:N1+N1]) * Y
line1,  = ax.plot(x1, y1, marker="o", linewidth=2, markersize=6, label = 'UAV_1')

x2 = np.array(population2[0][:N1]) * X
y2 = np.array(population2[0][N1:N1+N1]) * Y2
line2,  = ax.plot(x2, y2, marker="o", linewidth=2, markersize=6, label = 'UAV_2')

ax.legend()

ax.scatter(w_s[0], w_s[1], color="red", s=50)
ax.text(w_s[0], w_s[1], "Source", color="blue")
ax.scatter(w_d[0], w_d[1], color="blue", s=50)
ax.text(w_d[0], w_d[1], "Destination", color="red")

ax.scatter(q_I1[0], q_I1[1], color="green", s=50)
ax.text(q_I1[0], q_I1[1], "Start1", color="green")
ax.scatter(q_F1[0], q_F1[1], color="yellow", s=50)
ax.text(q_F1[0], q_F1[1], "Finish1", color="green")

ax.scatter(q_I2[0], q_I2[1], color="green", s=50)
ax.text(q_I2[0], q_I2[1], "Start2", color="green")
ax.scatter(q_F2[0], q_F2[1], color="yellow", s=50)
ax.text(q_F2[0], q_F2[1], "Finish2", color="green")

plt.show()

print("Result:", fitness1(population1[0], log=True) + fitness2(population2[0], log = True))
