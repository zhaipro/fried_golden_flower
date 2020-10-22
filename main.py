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
    y = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 9])
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


def func_v3(x, y):
    n = x.size
    r = 0
    for i in range(n):
        for j in range(n):
            if i > j:
                if x[i] > y[j]:
                    r += max(y[j], 2)
                elif x[i] < y[j]:
                    r += x[i]
                else:
                    r += y[j]
            elif i < j:
                if x[i] > y[j]:
                    r -= max(y[j], 2)
                elif x[i] < y[j]:
                    r -= x[i] + 1
                else:
                    r -= max(x[i], 2)
    return r


def func_v4(x, y):
    n = x.size
    r = 0
    for i in range(n):
        for j in range(n):
            if i > j:
                if x[i] > y[j]:
                    r += y[j] + max(y[j], 2) + 1
                else:
                    r += x[i] + max(x[i], 2)
            elif j > i:
                if y[j] > x[i]:
                    r -= x[i] + max(x[i], 2) + 1
                else:
                    r -= y[j] + max(y[j], 2)
    return r


def calc_v4(n=10, m=10):
    y = np.ones(n)
    y = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 10])
    x = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 10])
    for i in range(1000000):
        r = func_v4(x, y)
        # print(r)
        # exit()
        if r > 0:
            print('hahaha', i, r, x, y, flush=True)
            y = x
        # break
        x = np.random.randint(1, m + 1, n)
    print(r, y)


def calc_v45(n=10, m=10):
    y = np.ones(n)
    y = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 10])
    while True:
        x = np.ones(n)
        max_r = func_v4(x, y)
        for i in range(n):
            max_x = x[i]
            for j in range(1, m + 1):
                x[i] = j
                r = func_v4(x, y)
                if r > max_r:
                    max_r = r
                    max_x = j
            x[i] = max_x
        y = x
        print(max_r, x, flush=True)
        if max_r == 0:
            break


def func_v5(x, y):
    # y 庄家
    n = y.size
    r = 0
    for i in range(n):
        for j in range(n):
            if i > j:
                if 2 * x - 1 > y[j]:
                    r += max(1 + y[j] // 2 * 2, 3)
                elif 2 * x - 1 < y[j]:
                    r += 2 * x - 1
                else:
                    r += y[j]
            elif i < j:
                if 2 * x - 1 > y[j]:
                    r -= max(1 + y[j] // 2 * 2, 3)
                elif 2 * x - 1 < y[j]:
                    r -= x + 1
                else:
                    r -= max(x, 2)
    return r


def calc_v5(n=10, m=10):
    y = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 10])
    for x in range(1, 11):
        r = func_v5(x, y)
        print(r)


def calc_v35(n=15, m=10):
    x = np.ones(n)
    y = np.ones(n)
    for _ in range(10):
        max_r = func_v3(x, y)
        for i in range(n):
            max_x = x[i]
            for j in range(1, m + 1):
                x[i] = j
                r = func_v3(x, y)
                if r > max_r:
                    max_r = r
                    max_x = j
            x[i] = max_x
        print('x:', max_r, x, flush=True)

        min_r = func_v3(x, y)
        for i in range(n):
            min_y = y[i]
            for j in range(1, m + 1):
                y[i] = j
                r = func_v3(x, y)
                if r < min_r:
                    min_r = r
                    min_y = j
            y[i] = min_y

        print('y:', max_r, x, flush=True)


def func_v6(x, y):
    # y 庄家
    n, m = y.shape
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    q = x[i, j] * y[k, l]
                    if j == 0:
                        r -= q
                    elif l == 0:
                        r += q
                    elif i > k:
                        if j > l:
                            r += (l + 1) * q
                        else:
                            r += j * q
                    elif i < k:
                        if j > l:
                            r -= (l + 1) * q
                        else:
                            r -= (j + 1) * q
    return r


def calc_v6():
    n, m = 10, 10
    x = np.random.random((n, m))
    x /= x.sum(axis=1, keepdims=True)
    y = np.random.random((n, m))
    y /= y.sum(axis=1, keepdims=True)

    for i in range(20):
        max_r = func_v6(x, y)
        max_x = x
        for _ in range(100):
            r = func_v6(x, y)
            if r > max_r:
                max_r = r
                max_x = x
            x = np.random.random((n, m))
            x /= x.sum(axis=1, keepdims=True)
        x = max_x

        min_r = func_v6(x, y)
        min_y = y
        for _ in range(100):
            r = func_v6(x, y)
            if r < min_r:
                min_r = r
                min_y = y
            y = np.random.random((n, m))
            y /= y.sum(axis=1, keepdims=True)
        y = min_y
    # print(y)
    print(max_r, min_r)
    print(max_x)
    print(min_y)


# calc()
# savefig()
# calc_v2()
# calc_v3()
# calc_v35()
# calc_v4()
# calc_v45()
# calc_v5()
calc_v6()
# 关键是期望值，具体分布无关要紧？
