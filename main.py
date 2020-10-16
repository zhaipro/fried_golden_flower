import numpy as np
import matplotlib.pyplot as plt


def savefig():
    fn='result.jpg'
    y = np.array([0.001183, 0.01362867, 0.02068196, 0.00682253, 0.20850199,
                  0.27713576, 0.98135702, 0.69793358, 0.99663031, 0.98980148])
    n, = y.shape
    x = list(range(n))
    plt.plot(x, y, 'b', label='y')
    plt.title('fried golden flower')
    plt.xlabel('x')
    plt.ylabel('E(x)')
    plt.legend()
    plt.savefig(fn)


def func(a, b):
    n, = a.shape
    r = 0
    for pi in range(n):
        for pj in range(n):
            t = (a[pi] + b[pj]) / 2
            if pi > pj:
                r += t
            elif pi < pj:
                r -= t
    return r / n / n


def calc(n=10):
    max_a = np.array([0.001183, 0.01362867, 0.02068196, 0.00682253, 0.20850199,
                      0.27713576, 0.98135702, 0.69793358, 0.99663031, 0.98980148])
    # max_a = np.arange(n) / (n - 1)
    # max_a = np.random(n)
    for i in range(10000000):
        a = np.random.random(n)
        r = func(max_a, a)
        if r < 0:
            print('hahaha', i, a)
            max_b = max_a
            max_a = a
    print(max_a)
    print(max_b)
    print(func(max_a, max_b))
    return max_a


# calc()
savefig()
# 关键是期望值，具体分布无关要紧？
