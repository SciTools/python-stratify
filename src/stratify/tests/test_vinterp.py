import unittest

import dask.array as da
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

import stratify
import stratify._vinterp as vinterp


class IndexInterpolator(vinterp.PyFuncInterpolator):
    def interp_kernel(self, index, z_src, fz_src, level, output_array):
        output_array[:] = index


class DirectionExtrapolator(vinterp.PyFuncExtrapolator):
    def extrap_kernel(self, direction, z_src, fz_src, level, output_array):
        output_array[:] = np.inf if direction > 0 else -np.inf


class TestColumnInterpolation(unittest.TestCase):
    def interpolate(self, x_target, x_src, rising=None):
        x_target = np.array(x_target)
        x_src = np.array(x_src)
        fx_src = np.empty(x_src.shape)

        index_interp = IndexInterpolator()
        extrap_direct = DirectionExtrapolator()

        r1 = stratify.interpolate(
            x_target,
            x_src,
            fx_src,
            rising=rising,
            interpolation=index_interp,
            extrapolation=extrap_direct,
        )

        if rising is not None:
            r2 = stratify.interpolate(
                -1 * x_target,
                -1 * x_src,
                fx_src,
                rising=not rising,
                interpolation=index_interp,
                extrapolation=extrap_direct,
            )
            assert_array_equal(r1, r2)

        lazy_fx_src = da.asarray(fx_src, chunks=tuple(range(1, x_src.ndim + 1)))
        r3 = stratify.interpolate(
            x_target,
            x_src,
            lazy_fx_src,
            rising=rising,
            interpolation=index_interp,
            extrapolation=extrap_direct,
        )
        assert_array_equal(r1, r3.compute())

        return r1

    def test_interp_only(self):
        r = self.interpolate([1, 2, 3], [1, 3])
        assert_array_equal(r, [0, 1, 1])

    def test_interp_multi_level_single_source(self):
        r = self.interpolate([1.5, 2, 2.5], [1, 3])
        assert_array_equal(r, [1, 1, 1])

    def test_interp_single_level_multiple_source(self):
        r = self.interpolate([3.5], [1, 2, 3, 3, 4])
        assert_array_equal(r, [4])

    def test_lower_extrap_only(self):
        r = self.interpolate([1, 2, 3], [4, 5])
        assert_array_equal(r, [-np.inf, -np.inf, -np.inf])

    def test_upper_extrap_only(self):
        r = self.interpolate([1, 2, 3], [-4, -5], rising=True)
        assert_array_equal(r, [np.inf, np.inf, np.inf])

    def test_extrap_on_both_sides_only(self):
        r = self.interpolate([1, 2, 5, 6], [3, 4])
        assert_array_equal(r, [-np.inf, -np.inf, np.inf, np.inf])

    def test_interp_and_extrap(self):
        r = self.interpolate([1, 2, 3, 5, 6], [2, 4, 5])
        assert_array_equal(r, [-np.inf, 0, 1, 2, np.inf])

    def test_nan_in_target(self):
        msg = "The target coordinate .* NaN"
        with self.assertRaisesRegex(ValueError, msg):
            self.interpolate([1, np.nan], [2, 4, 5])

    def test_nan_in_src(self):
        msg = "The source coordinate .* NaN"
        with self.assertRaisesRegex(ValueError, msg):
            self.interpolate([1], [0, np.nan], rising=True)

    def test_all_nan_in_src(self):
        r = self.interpolate([1, 2, 3, 4], [np.nan, np.nan, np.nan])
        assert_array_equal(r, [np.nan, np.nan, np.nan, np.nan])

    def test_nan_in_src_not_a_problem(self):
        # If we pick levels low enough, we can get away with having NaNs
        # in the source.
        r = self.interpolate([1, 3], [2, 4, np.nan])
        assert_array_equal(r, [-np.inf, 1])

    def test_no_levels(self):
        r = self.interpolate([], [2, 4, np.nan])
        assert_array_equal(r, [])

    def test_wrong_rising_target(self):
        r = self.interpolate([2, 1], [1, 2])
        assert_array_equal(r, [1, np.inf])

    def test_wrong_rising_source(self):
        r = self.interpolate([1, 2], [2, 1], rising=True)
        assert_array_equal(r, [-np.inf, 0])

    def test_wrong_rising_source_and_target(self):
        # If we overshoot the first level, there is no hope,
        # so we end up extrapolating.
        r = self.interpolate([3, 2, 1, 0], [2, 1], rising=True)
        assert_array_equal(r, [np.inf, np.inf, np.inf, np.inf])

    def test_non_monotonic_coordinate_interp(self):
        result = self.interpolate([15, 5, 15.0], [10.0, 20, 0, 20])
        assert_array_equal(result, [1, 2, 3])

    def test_non_monotonic_coordinate_extrap(self):
        result = self.interpolate([0, 15, 16, 17, 5, 15.0, 25], [10.0, 40, 0, 20])
        assert_array_equal(result, [-np.inf, 1, 1, 1, 2, 3, np.inf])

    def test_length_one_interp(self):
        r = self.interpolate([1], [2], rising=True)
        assert_array_equal(r, [-np.inf])

    def test_auto_rising_not_enough_values(self):
        with self.assertRaises(ValueError):
            _ = self.interpolate([1], [2])

    def test_auto_rising_equal_values(self):
        # The code checks whether the first value is <= or equal to
        # the second. If it didn't, we'd end up with +inf, not -inf.
        r = self.interpolate([1], [2, 2])
        assert_array_equal(r, [-np.inf])


class Test_INTERPOLATE_LINEAR(unittest.TestCase):
    def interpolate(self, x_target):
        interpolation = stratify.INTERPOLATE_LINEAR
        extrapolation = DirectionExtrapolator()

        x_src = np.arange(5)
        fx_src = 10 * x_src

        # Use -2 to test negative number support.
        return stratify.interpolate(
            np.array(x_target) - 2,
            x_src - 2,
            fx_src,
            interpolation=interpolation,
            extrapolation=extrapolation,
        )

    def test_on_the_mark(self):
        assert_array_equal(self.interpolate([0, 1, 2, 3, 4]), [0, 10, 20, 30, 40])

    def test_zero_gradient(self):
        assert_array_equal(
            stratify.interpolate(
                [1], [0, 1, 1, 2], [10, 20, 30, 40], interpolation="linear"
            ),
            [20],
        )

    def test_inbetween(self):
        assert_array_equal(
            self.interpolate([0.5, 1.25, 2.5, 3.75]), [5, 12.5, 25, 37.5]
        )

    def test_high_precision(self):
        assert_array_almost_equal(
            self.interpolate([1.123456789]), [11.23456789], decimal=6
        )

    def test_single_point(self):
        # Test that a single input point that falls exactly on the target
        # level triggers a shortcut that avoids the expectation of >=2 source
        # points.
        interpolation = stratify.INTERPOLATE_LINEAR
        extrapolation = DirectionExtrapolator()

        r = stratify.interpolate(
            [2],
            [2],
            [20],
            interpolation=interpolation,
            extrapolation=extrapolation,
            rising=True,
        )
        self.assertEqual(r, 20)


class Test_INTERPOLATE_NEAREST(unittest.TestCase):
    def interpolate(self, x_target):
        interpolation = stratify.INTERPOLATE_NEAREST
        extrapolation = DirectionExtrapolator()

        x_src = np.arange(5)
        fx_src = 10 * x_src

        # Use -2 to test negative number support.
        return stratify.interpolate(
            np.array(x_target) - 2,
            x_src - 2,
            fx_src,
            interpolation=interpolation,
            extrapolation=extrapolation,
        )

    def test_on_the_mark(self):
        assert_array_equal(self.interpolate([0, 1, 2, 3, 4]), [0, 10, 20, 30, 40])

    def test_inbetween(self):
        # Nearest rounds down for exactly half way.
        assert_array_equal(self.interpolate([0.5, 1.25, 2.5, 3.75]), [0, 10, 20, 40])

    def test_high_precision(self):
        assert_array_equal(self.interpolate([1.123456789]), [10])


class Test_EXTRAPOLATE_NAN(unittest.TestCase):
    def interpolate(self, x_target):
        interpolation = IndexInterpolator()
        extrapolation = stratify.EXTRAPOLATE_NAN

        x_src = np.arange(5)
        fx_src = 10 * x_src

        # Use -2 to test negative number support.
        return stratify.interpolate(
            np.array(x_target) - 2,
            x_src - 2,
            fx_src,
            interpolation=interpolation,
            extrapolation=extrapolation,
        )

    def test_below(self):
        assert_array_equal(self.interpolate([-1]), [np.nan])

    def test_above(self):
        assert_array_equal(self.interpolate([5]), [np.nan])


class Test_EXTRAPOLATE_NEAREST(unittest.TestCase):
    def interpolate(self, x_target):
        interpolation = IndexInterpolator()
        extrapolation = stratify.EXTRAPOLATE_NEAREST

        x_src = np.arange(5)
        fx_src = 10 * x_src

        # Use -2 to test negative number support.
        return stratify.interpolate(
            np.array(x_target) - 2,
            x_src - 2,
            fx_src,
            interpolation=interpolation,
            extrapolation=extrapolation,
        )

    def test_below(self):
        assert_array_equal(self.interpolate([-1]), [0.0])

    def test_above(self):
        assert_array_equal(self.interpolate([5]), [40])


class Test_EXTRAPOLATE_LINEAR(unittest.TestCase):
    def interpolate(self, x_target):
        interpolation = IndexInterpolator()
        extrapolation = stratify.EXTRAPOLATE_LINEAR

        x_src = np.arange(5)
        # To spice things up a bit, let's make x_src non-equal distance.
        x_src[4] = 9
        fx_src = 10 * x_src

        # Use -2 to test negative number support.
        return stratify.interpolate(
            np.array(x_target) - 2,
            x_src - 2,
            fx_src,
            interpolation=interpolation,
            extrapolation=extrapolation,
        )

    def test_below(self):
        assert_array_equal(self.interpolate([-1]), [-10.0])

    def test_above(self):
        assert_array_almost_equal(self.interpolate([15.123]), [151.23])

    def test_zero_gradient(self):
        assert_array_almost_equal(
            stratify.interpolate([2], [0, 0], [1, 1], extrapolation="linear"),
            [1],
        )

    def test_npts(self):
        interpolation = IndexInterpolator()
        extrapolation = stratify.EXTRAPOLATE_LINEAR

        msg = r"Linear extrapolation requires at least 2 " r"source points. Got 1."
        with self.assertRaisesRegex(ValueError, msg):
            stratify.interpolate(
                [1, 3.0],
                [2],
                [20],
                interpolation=interpolation,
                extrapolation=extrapolation,
                rising=True,
            )


class Test_custom_extrap_kernel(unittest.TestCase):
    class my_kernel(vinterp.PyFuncExtrapolator):
        def __init__(self, *args, **kwargs):
            super(Test_custom_extrap_kernel.my_kernel, self).__init__(*args, **kwargs)

        def extrap_kernel(self, direction, z_src, fz_src, level, output_array):
            output_array[:] = -10

    def test(self):
        interpolation = IndexInterpolator()
        extrapolation = Test_custom_extrap_kernel.my_kernel()

        r = stratify.interpolate(
            [1, 3.0],
            [1, 2],
            [10, 20],
            interpolation=interpolation,
            extrapolation=extrapolation,
            rising=True,
        )
        assert_array_equal(r, [0, -10])


class Test_Interpolation(unittest.TestCase):
    def test_axis_m1(self):
        data = np.empty([5, 4, 23, 7, 3])
        zdata = np.empty([5, 4, 23, 7, 3])
        i = vinterp._Interpolation([1, 3], zdata, data)
        # 1288 == 5 * 4 * 23 * 7
        self.assertEqual(i._result_working_shape, (1, 3220, 2, 1))
        self.assertEqual(i.result_shape, (5, 4, 23, 7, 2))
        self.assertEqual(i._zp_reshaped.shape, (3220, 3, 1))
        self.assertEqual(i._fp_reshaped.shape, (1, 3220, 3, 1))
        self.assertEqual(i.axis, -1)
        self.assertEqual(i.orig_shape, data.shape)
        self.assertIsInstance(i.z_target, np.ndarray)
        self.assertEqual(list(i.z_target), [1, 3])

    def test_axis_0(self):
        data = zdata = np.empty([5, 4, 23, 7, 3])
        i = vinterp._Interpolation([1, 3], data, zdata, axis=0)
        # 1932 == 4 * 23 * 7 *3
        self.assertEqual(i._result_working_shape, (1, 1, 2, 1932))
        self.assertEqual(i.result_shape, (2, 4, 23, 7, 3))
        self.assertEqual(i._zp_reshaped.shape, (1, 5, 1932))

    def test_axis_2(self):
        data = zdata = np.empty([5, 4, 23, 7, 3])
        i = vinterp._Interpolation([1, 3], data, zdata, axis=2)
        # 1932 == 4 * 23 * 7 *3
        self.assertEqual(i._result_working_shape, (1, 20, 2, 21))
        self.assertEqual(i.result_shape, (5, 4, 2, 7, 3))
        self.assertEqual(i._zp_reshaped.shape, (20, 23, 21))

    def test_inconsistent_shape(self):
        data = np.empty([5, 4, 23, 7, 3])
        zdata = np.empty([5, 4, 3, 7, 3])
        emsg = "z_src .* is not a subset of fz_src"
        with self.assertRaisesRegex(ValueError, emsg):
            vinterp._Interpolation([1, 3], data, zdata, axis=2)

    def test_axis_out_of_bounds_fz_src_relative(self):
        # axis is out of bounds as identified by the absolute axis with z_src.
        data = np.empty((5, 4))
        zdata = np.empty((5, 4))
        axis = 4
        emsg = "Axis {} out of range"
        with self.assertRaisesRegex(ValueError, emsg.format(axis)):
            vinterp._Interpolation([1, 3], data, zdata, axis=axis)

    def test_axis_out_of_bounds_z_src_absolute(self):
        # axis is out of bounds as identified by the relative axis with fz_src.
        data = np.empty((5, 4))
        zdata = np.empty((3, 5, 4))
        axis = 0
        emsg = "Axis {} out of range"
        with self.assertRaisesRegex(ValueError, emsg.format(axis)):
            vinterp._Interpolation([1, 3], data, zdata, axis=axis)

    def test_axis_greater_than_z_src_ndim(self):
        # Ensure that axis is not unnecessarily constrained to the dimensions
        # of z_src.
        data = np.empty((4))
        zdata = np.empty((3, 5, 4))
        axis = 2
        result = vinterp._Interpolation(data.copy(), data, zdata, axis=axis)
        self.assertEqual(result.result_shape, (3, 5, 4))

    def test_nd_inconsistent_ndims(self):
        z_target = np.empty((2, 3, 4))
        z_src = np.empty((3, 4))
        fz_src = np.empty((2, 3, 4))
        emsg = "z_target and z_src must have the same number of dimensions"
        with self.assertRaisesRegex(ValueError, emsg):
            vinterp._Interpolation(z_target, z_src, fz_src)

    def test_nd_inconsistent_shape(self):
        z_target = np.empty((3, 2, 6))
        z_src = np.empty((3, 4, 5))
        fz_src = np.empty((2, 3, 4, 5))
        emsg = (
            "z_target and z_src have different shapes, "
            r"got \(3, :, 6\) != \(3, :, 5\)"
        )
        with self.assertRaisesRegex(ValueError, emsg):
            vinterp._Interpolation(z_target, z_src, fz_src, axis=2)

    def test_result_dtype_f4(self):
        interp = vinterp._Interpolation(
            [17.5], np.arange(4) * 10, np.arange(4, dtype="f4")
        )
        result = interp.interpolate()

        self.assertEqual(interp._target_dtype, np.dtype("f4"))
        self.assertEqual(result.dtype, np.dtype("f4"))

    def test_result_dtype_f8(self):
        interp = vinterp._Interpolation(
            [17.5], np.arange(4) * 10, np.arange(4, dtype="f8")
        )
        result = interp.interpolate()

        self.assertEqual(interp._target_dtype, np.dtype("f8"))
        self.assertEqual(result.dtype, np.dtype("f8"))


class Test__Interpolation_interpolate_z_target_nd(unittest.TestCase):
    def test_target_z_3d_on_axis_0(self):
        z_target = z_source = f_source = np.arange(3) * np.ones([4, 2, 3])
        interp = vinterp._Interpolation(
            z_target,
            z_source,
            f_source,
            axis=0,
            extrapolation=stratify.EXTRAPOLATE_NEAREST,
        )
        result = interp.interpolate_z_target_nd()
        assert_array_equal(result, f_source)

    def test_target_z_3d_on_axis_m1(self):
        z_target = z_source = f_source = np.arange(3) * np.ones([4, 2, 3])
        interp = vinterp._Interpolation(
            z_target,
            z_source,
            f_source,
            axis=-1,
            extrapolation=stratify.EXTRAPOLATE_NEAREST,
        )
        result = interp.interpolate_z_target_nd()
        assert_array_equal(result, f_source)

    def test_target_z_2d_over_3d_on_axis_1(self):
        """
        Test the case where z_target(2, 4) and z_src(3, 4) are 2d, but the
        source data fz_src(3, 3, 4) is 3d. z_target and z_src cover the last
        2 dimensions of fz_src. The axis of interpolation is axis=1 wrt fz_src.

        """
        # Generate the 3d source data fz_src(3, 3, 4)
        base = np.arange(3).reshape(1, 3, 1) * 2
        data = np.broadcast_to(base, (3, 3, 4))
        fz_src = data * np.arange(1, 4).reshape(3, 1, 1) * 10
        # Generate the 2d target coordinate z_target(2, 4)
        # The target coordinate is configured to request the interpolated
        # mid-points over axis=1 of fz_src.
        z_target = np.repeat(np.arange(1, 4, 2).reshape(2, 1), 4, axis=1) * 10
        # Generate the 2d source coordinate z_src(3, 4)
        z_src = np.repeat(np.arange(3).reshape(3, 1), 4, axis=1) * 20
        # Configure the vertical interpolator.
        interp = vinterp._Interpolation(z_target, z_src, fz_src, axis=1)
        # Perform the vertical interpolation.
        result = interp.interpolate_z_target_nd()
        # Generate the 3d expected interpolated result(3, 2, 4).
        expected = np.repeat(z_target[np.newaxis, ...], 3, axis=0)
        expected = expected * np.arange(1, 4).reshape(3, 1, 1)
        assert_array_equal(result, expected)

    def test_target_z_2d_over_3d_on_axis_m1(self):
        """
        Test the case where z_target(3, 3) and z_src(3, 4) are 2d, but the
        source data fz_src(3, 3, 4) is 3d. z_target and z_src cover the last
        2 dimensions of fz_src. The axis of interpolation is the default last
        dimension, axis=-1, wrt fx_src.

        """
        # Generate the 3d source data fz_src(3, 3, 4)
        base = np.arange(4) * 2
        data = np.broadcast_to(base, (3, 3, 4))
        fz_src = data * np.arange(1, 4).reshape(3, 1, 1) * 10
        # Generate the 2d target coordinate z_target(3, 3)
        # The target coordinate is configured to request the interpolated
        # mid-points over axis=-1 (aka axis=2) of fz_src.
        z_target = np.repeat(np.arange(1, 6, 2).reshape(1, 3), 3, axis=0) * 10
        # Generate the 2d source coordinate z_src(3, 4)
        z_src = np.repeat(np.arange(4).reshape(1, 4), 3, axis=0) * 20
        # Configure the vertical interpolator.
        interp = vinterp._Interpolation(
            z_target,
            z_src,
            fz_src,
        )
        # Perform the vertical interpolation.
        result = interp.interpolate_z_target_nd()
        # Generate the 3d expected interpolated result(3, 3, 3)
        expected = np.repeat(z_target[np.newaxis, ...], 3, axis=0)
        expected = expected * np.arange(1, 4).reshape(3, 1, 1)
        assert_array_equal(result, expected)


class Test_interpolate(unittest.TestCase):
    def test_target_z_3d_axis_0(self):
        z_target = z_source = f_source = np.arange(3) * np.ones([4, 2, 3])
        result = vinterp.interpolate(
            z_target, z_source, f_source, extrapolation="linear"
        )
        assert_array_equal(result, f_source)

    def test_dask(self):
        z_target = z_source = f_source = np.arange(3) * np.ones([4, 2, 3])
        reference = vinterp.interpolate(
            z_target, z_source, f_source, extrapolation="linear"
        )
        # Test with various combinations of lazy input
        f_src = da.asarray(f_source, chunks=(2, 1, 2))
        for z_tgt in (z_target, z_target.tolist(), da.asarray(z_target)):
            for z_src in (z_source, da.asarray(z_source)):
                result = vinterp.interpolate(
                    z_tgt, z_src, f_src, extrapolation="linear"
                )
                assert_array_equal(reference, result.compute())

    def test_dask_1d_target(self):
        z_target = np.array([0.5])
        z_source = f_source = np.arange(3) * np.ones([4, 2, 3])
        reference = vinterp.interpolate(
            z_target, z_source, f_source, axis=1, extrapolation="linear"
        )
        # Test with various combinations of lazy input
        f_src = da.asarray(f_source, chunks=(2, 1, 2))
        for z_tgt in (z_target, z_target.tolist(), da.asarray(z_target)):
            for z_src in (z_source, da.asarray(z_source)):
                result = vinterp.interpolate(
                    z_tgt, z_src, f_src, axis=1, extrapolation="linear"
                )
                assert_array_equal(reference, result.compute())


if __name__ == "__main__":
    unittest.main()
