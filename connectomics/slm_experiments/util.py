# Authors : Nicolas Poilvert, <nicolas.poilvert@gmail.com>
# Licence : BSD 3-clause


def get_reduced_tm(tm, h_size, w_size):

    assert h_size % 2 != 0
    assert w_size % 2 != 0

    assert h_size <= tm.shape[0]
    assert w_size <= tm.shape[1]

    h_start, h_stop = h_size / 2, tm.shape[0] - h_size / 2
    w_start, w_stop = w_size / 2, tm.shape[1] - w_size / 2

    new_tm = tm[h_start:h_stop, w_start:w_stop]

    return new_tm
