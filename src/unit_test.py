import numpy as np
from timeit import default_timer as timer
import procrutes

N = 10000

def opf(a):
    u, s, v = np.linalg.svd(a)
    r = np.matmul(u*np.array([1, 1, np.sign(np.linalg.det(np.matmul(u, v)))]), v)
    return r

def gen_rigid_rotation(n):
    R = np.random.randn(n, 3, 3).astype(np.float32)
    X = np.random.randn(n, 3, 10).astype(np.float32)*10
    Y = np.zeros((n, 3, 10), dtype=np.float32)

    for r, x, y in zip(R, X, Y):
        r[...] = opf(r)
        y[...] = np.matmul(r, x)

    return X, Y


def gen_streched_rotation(n):
    R = np.random.randn(n, 3, 3).astype(np.float32)
    S = np.exp(np.random.randn(n, 3, 1).astype(np.float32))
    X = np.random.randn(n, 3, 10).astype(np.float32)*10
    Y = np.zeros((n, 3, 10), dtype=np.float32)

    for r, s, x, y in zip(R, S, X, Y):
        r[...] = opf(r)
        y[...] = np.matmul(r, x*s)

    return X, Y


def get_func(method):
    if method == 'opf':
        func = procrutes.orthogonal_polar_factor
        stat_func = lambda a, r: (a, r,\
                np.linalg.norm(np.matmul(r.T, r)-np.eye(3)))
        data_func = lambda x: (np.random.randn(x, 3, 3).astype(np.float32),)
        print_fmt = 'matrix A:\n {}\nmatrix R: \n{}\nerror: {}'

    elif method == 'np_opf':
        func = opf
        stat_func = lambda a, r: (a, r,\
                np.linalg.norm(np.matmul(r.T, r)-np.eye(3)))
        data_func = lambda x: (np.random.randn(x, 3, 3).astype(np.float32),)
        print_fmt = 'matrix A:\n {}\nmatrix R: \n{}\nerror: {}'

    elif method == 'svd':
        func = procrutes.svd3
        stat_func = lambda a, usv: (a, *usv,\
                np.linalg.norm(np.matmul(usv[0]*usv[1], usv[2])-a))
        data_func = lambda x: (np.random.randn(x, 3, 3).astype(np.float32),)
        print_fmt = 'matrix A:\n{}\nmatrix U:\n{}\nmatrix S:\n{}\nmatrix Vh:\n{}\nerror: {}'

    elif method == 'procrutes':
        func = procrutes.procrutes
        data_func = gen_rigid_rotation
        stat_func = lambda x, y, o: (x, y, o, \
                np.linalg.norm(np.matmul(o, x) - y))
        print_fmt = 'matrix X:\n{}\nmatrix Y:\n{}\nmatrix R:\n{}\nerror: {}'

    elif method == 'anitropic_procrutes':
        func = procrutes.anitropic_procrutes
        data_func = gen_streched_rotation
        stat_func = lambda x, y, o: (x, y, *o, \
                np.linalg.norm(np.matmul(o[0].T, y)/o[1].reshape(3,1) - x))
        print_fmt = 'matrix X:\n{}\nmatrix Y:\n{}\nmatrix R:\n{}\nmatrix S:\n{}\nerror: {}'


    return data_func, func, stat_func, print_fmt


def test_correctness(data_func, func, stat_func, print_fmt):

    datas = data_func(1)
    for data in zip(*datas):
        out = stat_func(*data, func(*data))
        print(print_fmt.format(*out))
    return


def benchmark_accuracy(data_func, func, stat_func):

    datas = data_func(N)
    err_sum = 0
    for data in zip(*datas):
        err_sum += stat_func(*data, func(*data))[-1]

    print('Average error over {} random samples: {}'.format(N, err_sum/N))

    return


def benchmark_speed(data_func, func):

    datas = data_func(N)

    start = timer()
    for data in zip(*datas):
        _ = func(*data)
    end = timer()

    print('Average execution time over {} random samples: {} us'.format(N, (end-start)/N*1e6))

    return


def test_method_correctness(method):
    print('Testing method: {}'.format(method))
    funcs = get_func(method)
    test_correctness(*funcs)
    return


def benchmark_method_accuracy(method):
    print('Benchmarking accuracy: {}'.format(method))
    funcs = get_func(method)
    benchmark_accuracy(*funcs[:-1])
    return


def benchmark_method_speed(method):
    print('Benchmarking speed: {}'.format(method))
    funcs = get_func(method)
    benchmark_speed(*funcs[:-2])
    return


if  __name__ == '__main__':

    #test_method_correctness('opf')
    #test_method_correctness('svd')
    test_method_correctness('anitropic_procrutes')
    benchmark_method_accuracy('anitropic_procrutes')
    benchmark_method_speed('anitropic_procrutes')
    #benchmark_method_accuracy('opf')
    #benchmark_method_speed('opf')

