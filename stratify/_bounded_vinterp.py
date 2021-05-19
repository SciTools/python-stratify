import numpy as np

from ._conservative import conservative_interpolation


def interpolate_conservative(z_target, z_src, fz_src, axis=-1):
    """
    1d conservative interpolation across multiple dimensions.

    This function provides the ability to perform 1d interpolation on datasets
    with more than one dimension. For instance, this function can be used to
    interpolate a set of vertical levels, even if the interpolation coordinate
    depends upon other dimensions.

    A good use case might be when wanting to interpolate at a specific height
    for height data which also depends on x and y - e.g. extract 1000hPa level
    from a 3d dataset and associated pressure field. In the case of this
    example, pressure would be the `z` coordinate, and the dataset
    (e.g. geopotential height / temperature etc.) would be `f(z)`.

    Parameters
    ----------
    z_target: :class:`np.ndarray`
        Target coordinate.
        This coordinate defines the levels to interpolate the source data
        ``fz_src`` to.  ``z_target`` must have the same dimensionality as the
        source coordinate ``z_src``, and the shape of ``z_target`` must match
        the shape of ``z_src``, although the axis of interpolation may differ
        in dimension size.
    z_src: :class:`np.ndarray`
        Source coordinate.
        This coordinate defines the levels that the source data ``fz_src`` is
        interpolated from.
    fz_src: :class:`np.ndarray`
        The source data; the phenomenon data values to be interpolated from
        ``z_src`` to ``z_target``.
        The data array must be at least ``z_src.ndim``, and its trailing
        dimensions (i.e. those on its right hand side) must be exactly
        the same as the shape of ``z_src``.
    axis: int (default -1)
        The ``fz_src`` axis to perform the interpolation over.

    Returns
    -------
    : :class:`np.ndarray`
        fz_src interpolated from z_src to z_target.

    Note
    ----
    - Support for 1D z_target and corresponding ND z_src will be provided in
    future as driven by user requirement.
    - Those cells, where 'nan' values in the source data contribute, a 'nan'
      value is returned.

    """
    if z_src.ndim != z_target.ndim:
        msg = (
            "Expecting source and target levels dimensionality to be "
            "identical.  {} != {}."
        )
        raise ValueError(msg.format(z_src.ndim, z_target.ndim))

    # Relative axis
    axis = axis % fz_src.ndim
    axis_relative = axis - (fz_src.ndim - (z_target.ndim - 1))

    src_shape = list(z_src.shape)
    src_shape.pop(axis_relative)
    tgt_shape = list(z_target.shape)
    tgt_shape.pop(axis_relative)

    if src_shape != tgt_shape:
        src_shape = list(z_src.shape)
        src_shape[axis_relative] = "-"
        tgt_shape = list(z_target.shape)
        src_shape[axis_relative] = "-"
        msg = (
            "Expecting the shape of the source and target levels except "
            "the axis of interpolation to be identical.  {} != {}"
        )
        raise ValueError(msg.format(tuple(src_shape), tuple(tgt_shape)))

    dat_shape = list(fz_src.shape)
    dat_shape = dat_shape[-(z_src.ndim - 1) :]
    src_shape = list(z_src.shape[:-1])
    if dat_shape != src_shape:
        dat_shape = list(fz_src.shape)
        dat_shape[: -(z_src.ndim - 1)] = "-"
        msg = (
            "The provided data is not of compatible shape with the "
            "provided source bounds. {} != {}"
        )
        raise ValueError(msg.format(tuple(dat_shape), tuple(src_shape)))

    if z_src.shape[-1] != 2:
        msg = "Unexpected source and target bounds shape. shape[-1] != 2"
        raise ValueError(msg)

    # Define our source in a consistent way.
    # [broadcasting_dims, axis_interpolation, z_varying]

    # src_data
    bdims = list(range(fz_src.ndim - (z_src.ndim - 1)))
    data_vdims = [ind for ind in range(fz_src.ndim) if ind not in (bdims + [axis])]
    data_transpose = bdims + [axis] + data_vdims
    fz_src_reshaped = np.transpose(fz_src, data_transpose)
    fz_src_orig = list(fz_src_reshaped.shape)
    shape = (
        int(np.product(fz_src_reshaped.shape[: len(bdims)])),
        fz_src_reshaped.shape[len(bdims)],
        int(np.product(fz_src_reshaped.shape[len(bdims) + 1 :])),
    )
    fz_src_reshaped = fz_src_reshaped.reshape(shape)

    # Define our src and target bounds in a consistent way.
    # [axis_interpolation, z_varying, 2]
    vdims = list(set(range(z_src.ndim)) - set([axis_relative]))
    z_src_reshaped = np.transpose(z_src, [axis_relative] + vdims)
    z_target_reshaped = np.transpose(z_target, [axis_relative] + vdims)

    shape = int(np.product(z_src_reshaped.shape[1:-1]))
    z_src_reshaped = z_src_reshaped.reshape(
        [z_src_reshaped.shape[0], shape, z_src_reshaped.shape[-1]]
    )
    shape = int(np.product(z_target_reshaped.shape[1:-1]))
    z_target_reshaped = z_target_reshaped.reshape(
        [z_target_reshaped.shape[0], shape, z_target_reshaped.shape[-1]]
    )

    result = conservative_interpolation(
        z_src_reshaped, z_target_reshaped, fz_src_reshaped
    )

    # Turn the result into a shape consistent with the source.
    # First reshape, then reverse transpose.
    shape = fz_src_orig
    shape[len(bdims)] = z_target.shape[axis_relative]
    result = result.reshape(shape)
    invert_transpose = [data_transpose.index(ind) for ind in list(range(result.ndim))]
    result = result.transpose(invert_transpose)
    return result
