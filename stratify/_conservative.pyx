import numpy as np
cimport numpy as np


cdef calculate_weights(np.ndarray[np.float64_t, ndim=2] src_point,
                       np.ndarray[np.float64_t, ndim=2] tgt_point):
    # src_min    src_max
    #    |----------|                 : Source
    #       tgt_min     tgt_max
    #          |------------|         : Target
    #    |----------|                 : Delta (src_max - src_min)
    #          |----|                 : Overlap (between src & tgt)
    #    weight = overlap / delta
    cdef Py_ssize_t src_ind, tgt_ind
    cdef np.float64_t delta, weight
    cdef np.ndarray[np.float64_t, ndim = 2] weights
    cdef np.ndarray[np.float64_t, ndim = 1] src_cell, tgt_cell

    weights = np.zeros([src_point.shape[0], tgt_point.shape[0]],
                       dtype=np.float64)
    for src_ind, src_cell in enumerate(src_point):
        delta = src_cell.max() - src_cell.min()
        for tgt_ind, tgt_cell in enumerate(tgt_point):
            weight = (min(src_cell.max(), tgt_cell.max()) -
                      max(src_cell.min(), tgt_cell.min())) / float(delta)
            if weight > 0:
                weights[src_ind, tgt_ind] = weight
    return weights


cdef apply_weights(np.ndarray[np.float64_t, ndim=3] src_point,
                   np.ndarray[np.float64_t, ndim=3] tgt_point,
                   np.ndarray[np.float64_t, ndim=3] src_data):
    cdef Py_ssize_t ind
    cdef np.ndarray[np.float64_t, ndim = 3] results
    cdef np.ndarray[np.float64_t, ndim = 2] weights
    results = np.zeros(
        [src_data.shape[0], tgt_point.shape[0], src_data.shape[2]],
        dtype='float64')
    # Calculate and apply weights
    for ind in range(src_data.shape[2]):
        weights = calculate_weights(src_point[:, ind], tgt_point[:, ind])
        if not (weights.sum(axis=1) == 1).all():
            msg = ('Weights calculation yields a less than conservative '
                   'result.  Aborting.')
            raise ValueError(msg)
        results[..., ind] = (
            weights * src_data[..., ind][..., None]).sum(axis=1)
    return results


def conservative_interpolation(src_points, tgt_points, src_data):
    return apply_weights(src_points.astype('float64'),
                         tgt_points.astype('float64'),
                         src_data.astype('float64'))
