import numpy as np
cimport numpy as np


cdef calculate_weights(np.ndarray[np.float64_t, ndim=2] src_point,
                       np.ndarray[np.float64_t, ndim=2] tgt_point):
    """
    Calculate weights for a given point.

    The following visually illustrates the calculation::

        src_min    src_max
           |----------|                 : Source
              tgt_min     tgt_max
                 |------------|         : Target
           |----------|                 : Delta (src_max - src_min)
                 |----|                 : Overlap (between src & tgt)
           weight = overlap / delta

    Parameters
    ----------
    src_point (2d double array) - Source point (at a specific location).
    tgt_point (2d double array) - Target point (at a specific location).

    Returns
    -------
    2d double array - Weights corresponding to shape [src_point.shape[0],
        tgt_point.shape[0]].

    """
    cdef Py_ssize_t src_ind, tgt_ind
    cdef np.float64_t delta, weight
    cdef np.ndarray[np.float64_t, ndim=2] weights
    cdef np.ndarray[np.float64_t, ndim=1] src_cell, tgt_cell

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
    """
    Perform conservative interpolation.

    Conservation interpolation of a dataset between a provided source
    coordinate and a target coordinate.  Where no source cells contribute to a
    target cell, a np.nan value is returned.

    Parameters
    ----------
    src_points (3d double array) - Source coordinate, taking the form
        [axis_interpolation, z_varying, 2].
    tgt_points (3d double array) - Target coordinate, taking the form
        [axis_interpolation, z_varying, 2].
    src_data (3d double array) - The source data, the phenomenon data to be
        interpolated from ``src_points`` to ``tgt_points``.  Taking the form
        [broadcasting_dims, axis_interpolation, z_varying].

    Returns
    -------
    3d double array - Interpolated result over target levels (``tgt_points``).
        Taking the form [broadcasting_dims, axis_interpolation, z_varying].

    """
    cdef Py_ssize_t ind
    cdef np.ndarray[np.float64_t, ndim=3] results, weighted_contrib
    cdef np.ndarray[np.float64_t, ndim=2] weights
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
        weighted_contrib = weights * src_data[..., ind][..., None]
        results[..., ind] = (
            np.nansum(weighted_contrib, axis=1))
        # Return nan values for those target cells, where there is any
        # contribution of nan data from the source data.
        results[..., ind][
            ((weights > 0) * np.isnan(weighted_contrib)).any(axis=1)] = np.nan

        # Return np.nan for those target cells where no source contributes.
        results[:, weights.sum(axis=0) == 0, ind] = np.nan
    return results


def conservative_interpolation(src_points, tgt_points, src_data):
    """
    Perform conservative interpolation.

    Conservation interpolation of a dataset between a provided source
    coordinate and a target coordinate.  All inputs are recast to 64-bit float
    arrays before calculation.

    Parameters
    ----------
    src_points (3d array) - Source coordinate, taking the form
        [axis_interpolation, z_varying, 2].
    tgt_points (3d array) - Target coordinate, taking the form
        [axis_interpolation, z_varying, 2].
    src_data (3d array) - The source data, the phenonenon data to be
        interpolated from ``src_points`` to ``tgt_points``.  Taking the form
        [broadcasting_dims, axis_interpolation, z_varying].

    Returns
    -------
    3d double array - Interpolated result over target levels (``tgt_points``).
        Taking the form [broadcasting_dims, axis_interpolation, z_varying].

    """
    return apply_weights(src_points.astype('float64'),
                         tgt_points.astype('float64'),
                         src_data.astype('float64'))
