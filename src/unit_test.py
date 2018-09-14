import numpy as np
from timeit import default_timer as timer
import procrutes


def get_func(method):
    if method == 'opf':
        func = procrutes.orthogonal_polar_factor
        stat_func = lambda a, r: (a, r,\
                np.linalg.norm(np.matmul(r.T, r)-np.eye(3)))
        print_fmt = 'matrix A:\n {}\nmatrix R: \n{}\nerror: {}'

        return func, stat_func, print_fmt



def test_correctness(func, stat_func, print_fmt, data=None):
    if data is None:
        data = np.random.rand(3, 3).astype(np.float32)

    out = func(data)
    if type(out) == 'tuple':
        out = stat_func(data, *out)
    else:
        out = stat_func(data, out)
    print(print_fmt.format(*out))

    return data


def benchmark_accuracy(func, stat_func, data=None):
    if data is None:
        N = 100000
        data = np.random.rand(N, 3, 3).astype(np.float32)
    else:
        N = data.shape[0]

    err_sum = 0
    for A in data:
        out = func(A)
        if type(out) == 'tuple':
            out = stat_func(data, *out)
        else:
            out = stat_func(data, out)
        err_sum += out[-1]

    print('Average error over {} random samples: {}'.format(N, err_sum/N))

    return data


def benchmark_speed(func, data=None):
    if data is None:
        N = 100000
        data = np.random.rand(N, 3, 3).astype(np.float32)
    else:
        N = data.shape[0]

    start = timer()
    for A in data:
        _ = func(A)
    end = timer()

    print('Average execution time over {} random samples: {} us'.format(N, (end-start)/N*1e6))

    return data


def test_method_correctness(method, data=None):
    print('Testing method: {}'.format(method))
    funcs = get_func(method)
    return test_correctness(*funcs, data)


def benchmark_method_accuracy(method, data=None):
    print('Benchmarking accuracy: {}'.format(method))
    funcs = get_func(method)
    return benchmark_accuracy(*funcs[:-1], data)


def benchmark_method_speed(method, data=None):
    print('Benchmarking speed: {}'.format(method))
    funcs = get_func(method)
    return benchmark_speed(funcs[0], data)



if  __name__ == '__main__':

    test_method_correctness('opf')
    benchmark_method_accuracy('opf')
    benchmark_method_speed('opf')



