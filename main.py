import numpy as np
import matplotlib.pyplot as plt


def savefig():
    fn='result.jpg'
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                  1.0, 1.0, 1.0, 1.0, 1.0])
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
    max_a = np.array([0.0, 0.0, 0.0, 0.0, 0.0,
                      1.0, 1.0, 1.0, 1.0, 1.0])
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


def func_v2(x, y):
    n = x.size
    x.shape = -1, 1
    y.shape = 1, -1
    return np.sum(np.maximum((x > 1) * 2, np.minimum(x, y)) * (np.tril(np.ones((n, n))) - np.triu(np.ones((n, n)))))


def calc_v2(n=10, m=10):
    y = np.ones(n)
    # 1 庄家
    y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    x = np.array([1, 1, 1, 1, 1, 2, 4, 2, 7, 9])
    for i in range(1000000):
        r = func_v2(x, y) - func_v2(y, x)
        # print(r)
        # exit()
        if r > 0:
            print('hahaha', i, r, x, y.T, flush=True)
            y = x
        # break
        x = np.random.randint(1, m + 1, n)
    print(r, y.T)


# calc()
# savefig()
calc_v2()
# 关键是期望值，具体分布无关要紧？
