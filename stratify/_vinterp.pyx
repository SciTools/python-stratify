# Terminology:
# Z - the coordinate over which we are interpolating.
# z_src - the values of Z where fz_src is defined
# z_target - the desired values of Z to generate new data for.
# fz_src - the data, defined at each z_src
import functools

import numpy as np

cimport cython
cimport numpy as np


cdef extern from "numpy/npy_math.h" nogil:
    bint isnan "npy_isnan"(long double)
    float NAN "NPY_NAN"
    float INFINITY "NPY_INFINITY"


cdef extern from "math.h" nogil:
    double fabs(double z)


__all__ = ['interpolate', 'interp_schemes', 'extrap_schemes',
           'INTERPOLATE_LINEAR', 'INTERPOLATE_NEAREST',
           'EXTRAPOLATE_NAN', 'EXTRAPOLATE_NEAREST', 'EXTRAPOLATE_LINEAR',
           'PyFuncExtrapolator', 'PyFuncInterpolator']


cdef inline int relative_sign(double z, double z_base) nogil:
    """
    Return the sign of z relative to z_base.

    Parameters
    ----------
    z - the value to compare to z_base
    z_base - the other one

    Returns
    -------
    +1 if z > z_base, 0 if z == z_base, and -1 if z < z_base

    """
    cdef double delta

    delta = z - z_base
    # 1, -1, or 0. http://stackoverflow.com/a/1903975/741316
    return (delta > 0) - (delta < 0)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long gridwise_interpolation(double[:] z_target, double[:] z_src,
                                 double[:, :] fz_src, bint increasing,
                                 Interpolator interpolation,
                                 Extrapolator extrapolation,
                                 double [:, :] fz_target) nogil except -1:
    """
    Computes the interpolation of multiple levels of a single column.

    Parameters
    ----------
    z_target - the levels to interpolate the source data ``fz_src`` to.
    z_src - the levels that the source data ``fz_src`` is interpolated from.
    fz_src - the source data to be interpolated.
    increasing - true when increasing Z index generally implies increasing Z values
    interpolation - the inner interpolation functionality. See the definition of
                    Interpolator.
    extrapolation - the inner extrapolation functionality. See the definition of
                    Extrapolator.
    fz_target - the pre-allocated array to be used for the interpolated result

    Note: This algorithm is not symmetric. It does not make assumptions about monotonicity
          of z_src nor z_target. Instead, the algorithm marches forwards from the last
          z_target found. To visualise this, imagine a single vertical column of temperature
          values being our non-monotonic Z *coordinate*, with our fx data being height.
          At low indices of Z (i.e. at the bottom of the column) the temperature is high, and
          decreases as we ascend. As some point, the trend reverses, and the temperature
          again begins to rise, before finally trailing off again as we reach the very top
          of our column. Algorithmically, we march the column looking for the next z_target,
          when a crossing is detected we invoke the interpolation for fx at our current index.
          We then continue from this index, only looking for the crossing of the next z_target.

          For this reason, the order that the levels are provided is important.
          If z_src = [2, 4, 6], fz_src = [2, 4, 6] and z_target = [3, 5], fz_target will be
          [3, 5]. But if z_target = [5, 3], fz_target will be [5, <extrapolation value>].

    """
    cdef unsigned int i_src, i_target, n_src, n_target, i, m
    cdef bint all_nans = True
    cdef double z_before, z_current, z_after, z_last
    cdef int sign_after, sign_before, extrapolating

    n_src = z_src.shape[0]
    n_target = z_target.shape[0]

    # Check for a source coordinate that has only NaN values.
    if n_target and isnan(z_src[0]):
        for i in range(n_src):
            all_nans = isnan(z_src[i])
            if not all_nans:
                break
        if all_nans:
            # The result is also only NaN values.
            m = fz_target.shape[0]
            for i in range(m):
                for i_target in range(n_target):
                    fz_target[i, i_target] = NAN
            return 0

    interpolation.prepare_column(z_target, z_src, fz_src, increasing)
    extrapolation.prepare_column(z_target, z_src, fz_src, increasing)

    if increasing:
        z_before = -INFINITY
    else:
        z_before = INFINITY

    z_last = -z_before

    i_src = 0
    # The first window for which we are looking for a crossing is between the
    # first window value (typically -inf, but may be +inf) and the first z_src.
    # This search window will be moved along until a crossing is detected, at
    # which point we will do an interpolation.
    z_after = z_src[0]

    # We start in extrapolation mode. This will be turned off as soon as we
    # start increasing i_src.
    extrapolating = -1

    for i_target in range(n_target):
        # Move the level we are looking for forwards one.
        z_current = z_target[i_target]

        if isnan(z_current):
            with gil:
                raise ValueError('The target coordinate may not contain NaN values.')

        # Determine if the z_current has a crossing within
        # the current window.
        sign_before = relative_sign(z_before, z_current)
        sign_after = relative_sign(z_after, z_current)

        # Move the window forwards until a crossing *is* found for this level.
        # If we run out of z_src to check, we go back to extrapolation, and put the
        # upper edge of the window at z_last (typically +inf).
        while sign_before == sign_after:
            i_src += 1
            if i_src < n_src:
                extrapolating = 0
                z_after = z_src[i_src]
                if isnan(z_after):
                    with gil:
                        raise ValueError('The source coordinate may not contain NaN values.')
                sign_after = relative_sign(z_after, z_current)
            else:
                extrapolating = 1
                z_after = z_last
                sign_after = relative_sign(z_after, z_current)
                break

        if extrapolating == 0 or sign_after == 0:
            interpolation.kernel(i_src, z_src, fz_src, z_current,
                                 fz_target[:, i_target])
        else:
            extrapolation.kernel(extrapolating, z_src, fz_src, z_current,
                                 fz_target[:, i_target])

        # Move the lower edge of the window forwards to the level we've just computed,
        # thus preventing the levels from stepping back within a single index.
        z_before = z_current


cdef class Interpolator(object):
    cdef long kernel(self, unsigned int index,
                     double[:] z_src, double[:, :] fz_src,
                     double level, double[:] fz_level
                     ) nogil except -1:
        """
        The inner part of an interpolation operation.

        Parameters:
        ----------
        i (unsigned int) - the current (upper) index along z_src. 0 <= i < z_src.size[0]
                  i will only ever be 0 if z_src[i] == current_level.
                  the interpolation value may lie on exactly i, but will never lie on exactly i-1.
        z_src (double array) - the 1d column of z_src values.
        fz_src (2d double array) - the m 1d columns of fz_src values.
                               fz_src.shape[1] == z_src.shape[0].
                               fz_src.shape[0] may be 1 (common).
        current_level (double) - the value that we are interpolating for
        fz_target (double array) - the pre-allocated array to put the resulting
                                   interpolated values into.

        """
        with gil:
            raise RuntimeError('Interpolator subclasses should implement '
                               'the kernel function.')

    cdef bint prepare_column(self, double[:] z_target, double[:] z_src,
                      double[:, :] fz_src, bint increasing) nogil except -1:
        # Called before all levels are interpolated.
        pass


cdef class LinearInterpolator(Interpolator):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long kernel(self, unsigned int index, double[:] z_src, double[:, :] fz_src, double level,
                     double[:] fz_target) nogil except -1:
        """
        Compute a linear interpolation.

        The behaviour is as follows:
            * if we've found a crossing and it is exactly on the level, use the
              exact value from the original data (i.e. level == z_src[index])
            * otherwise, compute the distance of the level from i and i-1, and
              use these as proportions for linearly combining the fz_src values at
              those indices.

        """
        cdef unsigned int m = fz_src.shape[0]
        cdef double frac
        cdef unsigned int i

        if level == z_src[index]:
            for i in range(m):
                fz_target[i] = fz_src[i, index]
        else:
            frac = ((level - z_src[index - 1]) /
                    (z_src[index] - z_src[index - 1]))

            for i in range(m):
               fz_target[i] = fz_src[i, index - 1] + \
                                frac * (fz_src[i, index] - fz_src[i, index - 1])


cdef class NearestNInterpolator(Interpolator):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long kernel(self, unsigned int index, double[:] z_src, double[:, :] fz_src, double level,
                     double[:] fz_target) nogil except -1:
        """Compute a nearest-neighbour interpolation."""
        cdef unsigned int m = fz_src.shape[0]
        cdef unsigned int nearest_index, i

        if index != 0 and fabs(level - z_src[index - 1]) <= fabs(level - z_src[index]):
            nearest_index = index - 1
        else:
            nearest_index = index

        for i in range(m):
            fz_target[i] = fz_src[i, nearest_index]


cdef class PyFuncInterpolator(Interpolator):
    cdef bint use_column_prep

    def __init__(self, use_column_prep=True):
        self.use_column_prep = use_column_prep

    def column_prep(self, z_target, z_src, fz_src, increasing):
        """
        Called each time this interpolator sees a new data array.
        This method may be used for validation of a column, or for column
        based pre-interpolation calculations (e.g. spline gradients).

        Note: This method is not called if :attr:`.call_column_prep` is False.

        """
        pass

    cdef bint prepare_column(self, double[:] z_target, double[:] z_src,
                      double[:, :] fz_src, bint increasing) nogil except -1:
        if self.use_column_prep:
            with gil:
                self.column_prep(z_target, z_src, fz_src, increasing)

    def interp_kernel(self, index, z_src, fz_src, level, output_array):
        # Fill the output array with the fz_src data at the given index.
        # This is nealy equivalent to nearest neighbour, but doesn't take
        # into account which neighbour is nearest.
        output_array[:] = fz_src[:, index]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long kernel(self,
                     unsigned int index, double[:] z_src,
                     double[:, :] fz_src, double level,
                     double[:] fz_target) nogil except -1:
        with gil:
            self.interp_kernel(index, z_src, fz_src, level, fz_target)


cdef class Extrapolator(object):
    cdef long kernel(self, int direction,
                     double[:] z_src, double[:, :] fz_src,
                     double current_level, double[:] fz_target
                    ) nogil except -1:
        """
        Defines the inner part of an extrapolation operation.

        Parameters:
        ----------
        direction (int) - -1 for the bottom edge, +1 for the top edge
        z_src (double array) - the 1d column of z_src values.
        fz_src (2d double array) - the m 1d columns of fz_src values.
                               fz_src.shape[1] == z_src.shape[0].
                               fz_src.shape[0] may be 1 (common).
        current_level (double) - the value that we are interpolating for
        fz_target (double array) - the pre-allocated array to put the resulting
                                   extrapolated values into.
        """
        with gil:
            raise RuntimeError('Interpolator subclasses should implement '
                               'the kernel function.')

    cdef bint prepare_column(self, double[:] z_target, double[:] z_src,
                      double[:, :] fz_src, bint increasing) nogil except -1:
        pass


cdef class NaNExtrapolator(Extrapolator):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long kernel(self, int direction, double[:] z_src,
                     double[:, :] fz_src, double level,
                     double[:] fz_target) nogil except -1:
        """NaN values for extrapolation."""
        cdef unsigned int m = fz_src.shape[0]
        cdef unsigned int i

        for i in range(m):
            fz_target[i] = NAN


cdef class NearestNExtrapolator(Extrapolator):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long kernel(self,
                     int direction, double[:] z_src,
                     double[:, :] fz_src, double level,
                     double[:] fz_target) nogil except -1:
        """Nearest-neighbour/edge extrapolation."""
        cdef unsigned int m = fz_src.shape[0]
        cdef unsigned int index, i

        if direction < 0:
            index = 0
        else:
            index = fz_src.shape[1] - 1

        for i in range(m):
            fz_target[i] = fz_src[i, index]


cdef class LinearExtrapolator(Extrapolator):
    cdef bint prepare_column(self, double[:] z_target, double[:] z_src,
                      double[:, :] fz_src, bint increasing) nogil except -1:
        cdef unsigned int n_src_pts = z_src.shape[0]

        if n_src_pts < 2:
            with gil:
                raise ValueError('Linear extrapolation requires at least '
                                 '2 source points. Got {}.'.format(n_src_pts))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long kernel(self, int direction, double[:] z_src,
                     double[:, :] fz_src, double level,
                     double[:] fz_target) nogil except -1:
        """Linear extrapolation using either the first or last 2 values."""
        cdef unsigned int m = fz_src.shape[0]
        cdef unsigned int n_src_pts = fz_src.shape[1]
        cdef unsigned int p0, p1, i
        cdef double frac, z_step

        if direction < 0:
            p0, p1 = 0, 1
        else:
            p0, p1 = n_src_pts - 2, n_src_pts - 1

        # Compute the normalised distance of the target point between p0 and p1
        z_step = z_src[p1] - z_src[p0]
        if z_step == 0:
            # If there is nothing between the last two points then we
            # extrapolate using a 0 gradient.
            frac = 0
        else:
            frac = ((level - z_src[p0]) / z_step)

        for i in range(m):
           fz_target[i] = fz_src[i, p0] + frac * (fz_src[i, p1] - fz_src[i, p0])


cdef class PyFuncExtrapolator(Extrapolator):
    cdef bint use_column_prep

    def __init__(self, use_column_prep=True):
        self.use_column_prep = use_column_prep

    def column_prep(self, z_target, z_src, fz_src, increasing):
        """
        Called each time this extrapolator sees a new data array.
        This method may be used for validation of a column, or for column
        based pre-interpolation calculations (e.g. spline gradients).

        Note: This method is not called if :attr:`.call_column_prep` is False.

        """
        pass

    cdef bint prepare_column(self, double[:] z_target, double[:] z_src,
                      double[:, :] fz_src, bint increasing) nogil except -1:
        if self.use_column_prep:
            with gil:
                self.column_prep(z_target, z_src, fz_src, increasing)

    def extrap_kernel(self, direction, z_src, fz_src, level, output_array):
        # Fill the output array with nans.
        output_array[:] = np.nan

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef long kernel(self,
                     int direction, double[:] z_src,
                     double[:, :] fz_src, double level,
                     double[:] fz_target) nogil except -1:
        with gil:
            self.extrap_kernel(direction, z_src, fz_src, level, fz_target)


interp_schemes = {'nearest': NearestNInterpolator,
                  'linear': LinearInterpolator}

extrap_schemes = {'nearest': NearestNExtrapolator,
                  'linear': LinearExtrapolator,
                  'nan': NaNExtrapolator}


# Construct interp/extrap constants exposed to the user.
INTERPOLATE_LINEAR = interp_schemes['linear']()
INTERPOLATE_NEAREST = interp_schemes['nearest']()
EXTRAPOLATE_NAN = extrap_schemes['nan']()
EXTRAPOLATE_NEAREST = extrap_schemes['nearest']()
EXTRAPOLATE_LINEAR = extrap_schemes['linear']()


def interpolate(z_target, z_src, fz_src, axis=-1, rising=None,
                interpolation='linear', extrapolation='nan'):
    """
    Interface for optimised 1d interpolation across multiple dimensions.

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
    z_target: 1d or nd array
        Target coordinate.
        This coordinate defines the levels to interpolate the source data
        ``fz_src`` to. If ``z_target`` is an nd array, it must have the same
        dimensionality as the source coordinate ``z_src``, and the shape of
        ``z_target`` must match the shape of ``z_src``, although the axis
        of interpolation may differ in dimension size.
    z_src: nd array
        Source coordinate.
        This coordinate defines the levels that the source data ``fz_src`` is
        interpolated from.
    fz_src: nd array
        The source data; the phenomenon data values to be interpolated from
        ``z_src`` to ``z_target``.
        The data array must be at least ``z_src.ndim``, and its trailing
        dimensions (i.e. those on its right hand side) must be exactly
        the same as the shape of ``z_src``.
    axis: int (default -1)
        The ``fz_src`` axis to perform the interpolation over.
    rising: bool (default None)
        Whether the values of the source's interpolation coordinate values
        are generally rising or generally falling. For example, values of
        pressure levels will be generally falling as the z coordinate
        increases.
        This will determine whether extrapolation needs to occur for
        ``z_target`` below the first and above the last ``z_src``.
        If rising is None, the first two interpolation coordinate values
        will be used to determine the general direction. In most cases,
        this is a good option.
    interpolation: :class:`.Interpolator` instance or valid scheme name
        The core interpolation operation to use. :attr:`.INTERPOLATE_LINEAR`
        and :attr:`_INTERPOLATE_NEAREST` are provided for convenient
        iterpolation modes. Linear interpolation is the default.
    extrapolation: :class:`.Extrapolator` instance or valid scheme name
        The core extrapolation operation to use. :attr:`.EXTRAPOLATE_NAN` and
        :attr:`.EXTRAPOLATE_NEAREST` are provided for convenient extrapolation
        modes. NaN extrapolation is the default.

    """
    func = functools.partial(
        _interpolate,
        axis=axis,
        rising=rising,
        interpolation=interpolation,
        extrapolation=extrapolation
    )
    if not hasattr(fz_src, 'compute'):
        # Numpy array
        return func(z_target, z_src, fz_src)

    # Dask array
    import dask.array as da

    # Ensure z_target is an array.
    if not isinstance(z_target, (np.ndarray, da.Array)):
      z_target = np.array(z_target)

    # Compute chunk sizes
    if axis < 0:
      axis += fz_src.ndim
    in_chunks = list(fz_src.chunks)
    in_chunks[axis] = fz_src.shape[axis]

    out_chunks = list(in_chunks)
    if z_target.ndim == 1:
        out_chunks[axis] = z_target.shape[0]
    else:
        out_chunks[axis] = z_target.shape[axis]

    # Ensure `fz_src` is not chunked along `axis`.
    fz_src = fz_src.rechunk(in_chunks)

    # Ensure z_src is a dask array with the correct chunks.
    if isinstance(z_src, da.Array):
        z_src = z_src.rechunk(in_chunks)
    else:
        z_src = da.asarray(z_src, chunks=in_chunks)

    # Compute with 1-dimensional target array.
    if z_target.ndim == 1:
        func = functools.partial(func, z_target)
        return da.map_blocks(func, z_src, fz_src,
                             chunks=out_chunks, dtype=fz_src.dtype,
                             meta=np.array((), dtype=fz_src.dtype))

    # Ensure z_target is a dask array with the correct chunks
    if isinstance(z_target, da.Array):
       z_target = z_target.rechunk(out_chunks)
    else:
       z_target = da.asarray(z_target, chunks=out_chunks)

    # Compute with multi-dimensional target array.
    return da.map_blocks(func, z_target, z_src, fz_src,
                         chunks=out_chunks, dtype=fz_src.dtype,
                         meta=np.array((), dtype=fz_src.dtype))


def _interpolate(z_target, z_src, fz_src, axis=-1, rising=None,
                 interpolation='linear', extrapolation='nan'):
    if interpolation in interp_schemes:
        interpolation = interp_schemes[interpolation]()
    if extrapolation in extrap_schemes:
        extrapolation = extrap_schemes[extrapolation]()

    interp = _Interpolation(z_target, z_src, fz_src, rising=rising, axis=axis,
                            interpolation=interpolation,
                            extrapolation=extrapolation)
    if interp.z_target.ndim == 1:
        return interp.interpolate()
    else:
        return interp.interpolate_z_target_nd()


cdef class _Interpolation(object):
    """
    Where the magic happens for gridwise_interp. The work of this __init__ is
    mostly for putting the input nd arrays into a 3 and 4 dimensional form for
    convenient (read: efficient) Cython form. Inline comments should help with
    understanding.

   """
    cdef Interpolator interpolation
    cdef Extrapolator extrapolation

    cdef public np.dtype _target_dtype
    cdef int rising
    cdef public z_target, orig_shape, axis, _zp_reshaped, _fp_reshaped
    cdef public _result_working_shape, result_shape

    def __init__(self, z_target, z_src, fz_src, axis=-1,
                 rising=None,
                 Interpolator interpolation=INTERPOLATE_LINEAR,
                 Extrapolator extrapolation=EXTRAPOLATE_NAN):
        # Cast data to numpy arrays if not already.
        z_target = np.array(z_target, dtype=np.float64)
        z_src = np.array(z_src, dtype=np.float64)
        fz_src = np.array(fz_src)
        #: The result data dtype.
        if np.issubdtype(fz_src.dtype, np.signedinteger):
            self._target_dtype = np.dtype('f8')
        else:
            self._target_dtype = fz_src.dtype
        fz_src = fz_src.astype(np.float64)

        # Compute the axis in absolute terms.
        fp_axis = (axis + fz_src.ndim) % fz_src.ndim
        zp_axis = fp_axis - (fz_src.ndim - z_src.ndim)
        if (not 0 <= zp_axis < z_src.ndim) or (axis >= fz_src.ndim):
            raise ValueError('Axis {} out of range.'.format(axis))

        # Ensure that fz_src's shape is a superset of z_src's.
        if z_src.shape != fz_src.shape[-z_src.ndim:]:
            emsg = 'Shape for z_src {} is not a subset of fz_src {}.'
            raise ValueError(emsg.format(z_src.shape, fz_src.shape))

        if z_target.ndim == 1:
            z_target_size = z_target.shape[0]
        else:
            # Ensure z_target and z_src have same ndims.
            if z_target.ndim != z_src.ndim:
                emsg = ('z_target and z_src must have the same number '
                        'of dimensions, got {} != {}.')
                raise ValueError(emsg.format(z_target.ndim, z_src.ndim))
            # Ensure z_target and z_src have same shape over their
            # non-interpolated axes i.e. we need to ignore the axis of
            # interpolation when comparing the shapes of z_target and z_src.
            # E.g a z_target.shape=(3, 4, 5) and z_src.shape=(3, 10, 5),
            # interpolating over zp_axis=1 is fine as (3, :, 5) == (3, :, 5).
            # However, a z_target.shape=(3, 4, 6) and z_src.shape=(3, 10, 5),
            # interpolating over zp_axis=1 must fail as (3, :, 6) != (3, :, 5)
            zts, zss = z_target.shape, z_src.shape
            ztsp, zssp = zip(*[(str(j), str(k)) if i!=zp_axis else (':', ':')
                               for i, (j, k) in enumerate(zip(zts, zss))])
            if ztsp != zssp:
                sep, emsg = ', ', ('z_target and z_src have different shapes, '
                                   'got ({}) != ({}).')
                raise ValueError(emsg.format(sep.join(ztsp), sep.join(zssp)))
            z_target_size = zts[zp_axis]

        # We are going to put the source coordinate into a 3d shape for convenience of
        # Cython interface. Writing generic, fast, n-dimensional Cython code
        # is not possible, but it is possible to always support a 3d array with
        # the middle dimensions being from the axis argument.

        # Work out the shape of the left hand side of the 3d array.
        lh_dim_size = ([1] + list(np.cumprod(z_src.shape)))[zp_axis]

        # The coordinate shape will therefore be (size of lhs, size of axis, the rest).
        new_shape = (lh_dim_size, z_src.shape[zp_axis], -1)

        #: The levels to interpolate onto.
        self.z_target = z_target
        #: The shape of the input data (fz_src).
        self.orig_shape = fz_src.shape
        #: The fz_src axis over which to do the interpolation.
        self.axis = axis

        #: The source z coordinate data reshaped into 3d working shape form.
        self._zp_reshaped = z_src.reshape(new_shape)
        #: The fz_src data reshaped into 4d working shape form. The left-most
        #: dimension is the dimension broadcast dimension - these are the
        #: values which are not dependent on z, and come from the fact that
        #: fz_src may be higher dimensional than z_src.
        self._fp_reshaped = fz_src.reshape((-1, ) + self._zp_reshaped.shape)

        # Figure out the normalised 4d shape of the working result array.
        # This will be the same as _fp_reshaped, but the length of the
        # interpolation axis will change.
        result_working_shape = list(self._fp_reshaped.shape)
        # Update the axis to be of the size of the levels.
        result_working_shape[2] = z_target_size

        #: The shape of result while the interpolation is being calculated.
        self._result_working_shape = tuple(result_working_shape)

        # Figure out the nd shape of the result array.
        # This will be the same as fz_src, but the length of the
        # interpolation axis will change.
        result_shape = list(self.orig_shape)
        # Update the axis to be of the size of the levels.
        result_shape[fp_axis] = z_target_size

        #: The shape of the interpolated data.
        self.result_shape = tuple(result_shape)

        if rising is None:
            if z_src.shape[zp_axis] < 2:
                raise ValueError('The rising keyword must be defined when '
                                 'the size of the source array is <2 in '
                                 'the interpolation axis.')
            z_src_indexer = [0] * z_src.ndim
            z_src_indexer[zp_axis] = slice(0, 2)
            first_two = z_src[tuple(z_src_indexer)]
            rising = first_two[0] <= first_two[1]

        self.rising = bool(rising)

        # Sometimes we want to add additional constraints on our interpolation
        # and extrapolation - for example, linear extrapolation requires there
        # to be two coordinates to interpolate from.
        if hasattr(self.interpolation, 'validate_data'):
            self.interpolation.validate_data(self)

        if hasattr(self.extrapolation, 'validate_data'):
            self.extrapolation.validate_data(self)

        self.interpolation = interpolation
        self.extrapolation = extrapolation

    def interpolate(self):
        # Construct the output array for the interpolation to fill in.
        fz_target = np.empty(self._result_working_shape, dtype=np.float64)

        cdef unsigned int i, j, ni, nj

        ni = fz_target.shape[1]
        nj = fz_target.shape[3]

        # Pull in our pre-formed z_target, z_src, and fz_src arrays.
        cdef double[:] z_target = self.z_target
        cdef double[:, :, :] z_src = self._zp_reshaped
        cdef double[:, :, :, :] fz_src = self._fp_reshaped

        # Construct a memory view of the fz_target array.
        cdef double[:, :, :, :] fz_target_view = fz_target

        # Release the GIL and do the for loop over the left-hand, and
        # right-hand dimensions. The loop optimised for row-major data (C).
        with nogil:
            for j in range(nj):
                for i in range(ni):
                    gridwise_interpolation(z_target, z_src[i, :, j], fz_src[:, i, :, j],
                                           self.rising,
                                           self.interpolation,
                                           self.extrapolation,
                                           fz_target_view[:, i, :, j])
        return fz_target.reshape(self.result_shape).astype(self._target_dtype)

    def interpolate_z_target_nd(self):
        """
        Using the exact same functionality as found in interpolate, only without the assumption
        that the target z is 1d.

        """
        # Construct the output array for the interpolation to fill in.
        fz_target = np.empty(self._result_working_shape, dtype=np.float64)

        cdef unsigned int i, j, ni, nj

        ni = fz_target.shape[1]
        nj = fz_target.shape[3]

        z_target_reshaped = self.z_target.reshape(self._result_working_shape[1:])
        cdef double[:, :, :] z_target = z_target_reshaped

        # Pull in our pre-formed z_src, and fz_src arrays.
        cdef double[:, :, :] z_src = self._zp_reshaped
        cdef double[:, :, :, :] fz_src = self._fp_reshaped

        # Construct a memory view of the fz_target array.
        cdef double[:, :, :, :] fz_target_view = fz_target

        # Release the GIL and do the for loop over the left-hand, and
        # right-hand dimensions. The loop optimised for row-major data (C).
        with nogil:
            for j in range(nj):
                for i in range(ni):
                    gridwise_interpolation(z_target[i, :, j], z_src[i, :, j], fz_src[:, i, :, j],
                                           self.rising,
                                           self.interpolation,
                                           self.extrapolation,
                                           fz_target_view[:, i, :, j])

        return fz_target.reshape(self.result_shape).astype(self._target_dtype)
