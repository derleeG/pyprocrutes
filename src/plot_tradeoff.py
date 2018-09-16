import matplotlib.pyplot as plt
from unit_test import *

if __name__ == '__main__':


    N = 1000
    data_func, ours_func, stat_func, _ = get_func('anitropic_procrutes')
    _, np_func, _, _ = get_func('np_anitropic_procrutes')

    ours_err, ours_time = [], []
    np_err, np_time = [], []

    datas = data_func(N)

    for it in range(1, 60, 2):
        ours_err_sum, np_err_sum = 0, 0
        for data in zip(*datas):
            ours_err_sum += stat_func(*data, ours_func(*data, iter_num=it))[-1]
            np_err_sum += stat_func(*data, np_func(*data, iter_num=it))[-1]

        ours_err.append(ours_err_sum/N)
        np_err.append(np_err_sum/N)

        start = timer()
        for data in zip(*datas):
            _ = ours_func(*data, iter_num=it)
        end = timer()

        ours_time.append((end-start)/N*1e6)

        start = timer()
        for data in zip(*datas):
            _ = np_func(*data, iter_num=it)
        end = timer()

        np_time.append((end-start)/N*1e6)

        print('iter_num: {}, ours error: {:.3E}, ours time: {:.3f}us, np error: {:.3E}, np time: {:.3f}us'\
                .format(it, ours_err[-1], ours_time[-1], np_err[-1], np_time[-1]))


    ours_err = np.array(ours_err)
    ours_time = np.array(ours_time)
    np_err = np.array(np_err)
    np_time = np.array(np_time)
    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(ours_time, ours_err, marker='^', label='pyprocrutes')
    plt.scatter(np_time, np_err, marker='o', label='numpy')
    plt.xlabel("execution time (us)")
    plt.ylabel("error (||RSX-Y||_F/||Y||_F)")
    plt.legend()
    plt.show()



