import unittest

import numpy as np
from numpy.testing import assert_array_equal

import stratify._bounded_vinterp as bounded_vinterp


class Test1D(unittest.TestCase):
    # 1D cases represent the situation where the vertical coordinate is
    # 1-dimensional and as such, there are no dimension which the vertical
    # coordinates vary over other than the 'axis of interpolation'.
    # The common approach in solving these problems is to reshape and
    # transpose arrays into the form:
    # [broadcasting_dims, axis_interpolation, z_varying].
    # This this will take the form:
    # [broadcasting_dims, axis_interpolation, 1].
    def setUp(self):
        self.data = np.ones((4, 6))
        self.bounds = self.gen_bounds(0, 7, 1)

    def gen_bounds(self, start, stop, step):
        bounds = np.vstack(
            [
                np.arange(start, stop - step, step),
                np.arange(start + step, stop, step),
            ]
        )
        bounds = bounds.transpose((1, 0))
        return bounds.copy()

    def test_target_half_resolution(self):
        target_bounds = self.gen_bounds(0, 7, 2)
        res = bounded_vinterp.interpolate_conservative(
            target_bounds, self.bounds, self.data, axis=1
        )
        target_data = np.ones((4, 3)) * 2
        assert_array_equal(res, target_data)

    def test_target_double_resolution(self):
        target_bounds = self.gen_bounds(0, 6.5, 0.5)
        res = bounded_vinterp.interpolate_conservative(
            target_bounds, self.bounds, self.data, axis=1
        )
        target_data = np.ones((4, 12)) * 0.5
        assert_array_equal(res, target_data)

    def test_no_broadcasting(self):
        # In the case of no broadcasting, transposed and reshaped array of form
        # [broadcasting_dims, axis_interpolation, z_varying] should end up
        # [1, axis_interpolation, 1].
        data = self.data[0]
        target_bounds = self.gen_bounds(0, 7, 2)
        res = bounded_vinterp.interpolate_conservative(
            target_bounds, self.bounds, data, axis=0
        )
        target_data = np.ones((3)) * 2
        assert_array_equal(res, target_data)

    def test_source_with_nans(self):
        # In the case of no broadcasting, transposed and reshaped array of form
        # [broadcasting_dims, axis_interpolation, z_varying] should end up
        # [1, axis_interpolation, 1].
        data = self.data[0]
        data[0] = data[-2:] = np.nan
        target_bounds = self.gen_bounds(0, 6.5, 0.5)
        res = bounded_vinterp.interpolate_conservative(
            target_bounds, self.bounds, data, axis=0
        )
        target_data = np.ones((12)) * 0.5
        target_data[:2] = np.nan
        target_data[-4:] = np.nan
        assert_array_equal(res, target_data)

    def test_target_extends_above_source(self):
        # |-|-|-|-|-|-|-|   - Source
        # |-|-|-|-|-|-|-|-| - Target
        source_bounds = self.gen_bounds(0, 7, 1)
        target_bounds = self.gen_bounds(0, 8, 1)
        res = bounded_vinterp.interpolate_conservative(
            target_bounds, source_bounds, self.data, axis=1
        )
        target_data = np.ones((4, 7))
        target_data[:, -1] = np.nan
        assert_array_equal(res, target_data)

    def test_target_extends_above_source_non_equally_spaced_coords(self):
        # |--|--|-------||  - Source
        # |-|-|-|-|-|-|-|-| - Target
        source_bounds = np.array([[0, 1.5], [1.5, 2], [2, 6], [6, 6.5]])
        target_bounds = self.gen_bounds(0, 8, 1)
        data = np.ones((4, 4))
        res = bounded_vinterp.interpolate_conservative(
            target_bounds, source_bounds, data, axis=1
        )
        target_data = np.array(
            [1 / 1.5, 1 + ((1 / 3.0) / 1), 0.25, 0.25, 0.25, 0.25, 1.0]
        )[None]
        target_data = np.repeat(target_data, 4, 0)
        assert_array_equal(res, target_data)

    def test_target_extends_below_source(self):
        #   |-|-|-|-|-|-|-|   - Source
        # |-|-|-|-|-|-|-|-|   - Target
        source_bounds = self.gen_bounds(0, 7, 1)
        target_bounds = self.gen_bounds(-1, 7, 1)
        res = bounded_vinterp.interpolate_conservative(
            target_bounds, source_bounds, self.data, axis=1
        )
        target_data = np.ones((4, 7))
        target_data[:, 0] = np.nan
        assert_array_equal(res, target_data)


class TestND(unittest.TestCase):
    # ND cases represent the situation where the vertical coordinate varies
    # over dimensions other than the axis of interpolation.
    def setUp(self):
        self.data = np.ones((2, 6, 4, 3))
        self.bounds = self.gen_bounds(0, 7, 1)

    def gen_bounds(self, start, stop, step):
        bounds = np.vstack(
            [
                np.arange(start, stop - step, step),
                np.arange(start + step, stop, step),
            ]
        )
        bounds = bounds.transpose((1, 0))
        bounds = bounds[..., None, :].repeat(4, -2)
        bounds = bounds[..., None, :].repeat(3, -2)
        return bounds.copy()

    def test_target_half_resolution(self):
        target_bounds = self.gen_bounds(0, 7, 2)
        res = bounded_vinterp.interpolate_conservative(
            target_bounds, self.bounds, self.data, axis=1
        )

        target_data = np.ones((2, 3, 4, 3)) * 2
        assert_array_equal(res, target_data)

    def test_target_half_resolution_alt_axis(self):
        # Ensure results as expected with an alternative axis of interpolation.
        data = self.data.transpose((0, 2, 1, 3))
        bounds = self.bounds.transpose((1, 0, 2, 3))
        target_bounds = self.gen_bounds(0, 7, 2)
        target_bounds = target_bounds.transpose((1, 0, 2, 3))

        res = bounded_vinterp.interpolate_conservative(
            target_bounds, bounds, data, axis=2
        )
        target_data = np.ones((2, 4, 3, 3)) * 2
        assert_array_equal(res, target_data)

    def test_target_double_resolution(self):
        target_bounds = self.gen_bounds(0, 6.5, 0.5)
        res = bounded_vinterp.interpolate_conservative(
            target_bounds, self.bounds, self.data, axis=1
        )
        target_data = np.ones((2, 12, 4, 3)) * 0.5
        assert_array_equal(res, target_data)


class TestExceptions(unittest.TestCase):
    def test_mismatch_source_target_level_dimensionality(self):
        source_bounds = np.zeros((3, 4, 2))
        target_bounds = np.zeros((4, 2))
        data = np.zeros((3, 4))

        msg = "Expecting source and target levels dimensionality"
        with self.assertRaisesRegex(ValueError, msg):
            bounded_vinterp.interpolate_conservative(target_bounds, source_bounds, data)

    def test_mismatch_source_target_level_shape(self):
        # The source and target levels should have identical shape, other than
        # the axis of interpolation.
        source_bounds = np.zeros((3, 4, 2))
        target_bounds = np.zeros((2, 5, 2))
        data = np.zeros((3, 4))

        msg = (
            "Expecting the shape of the source and target levels except "
            "the axis of interpolation to be identical.  "
            r"\('-', 4, 2\) != \(2, 5, 2\)"
        )
        with self.assertRaisesRegex(ValueError, msg):
            bounded_vinterp.interpolate_conservative(
                target_bounds, source_bounds, data, axis=0
            )

    def test_mismatch_between_source_levels_source_data(self):
        # The source levels should reflect the shape of the data.
        source_bounds = np.zeros((2, 4, 2))
        target_bounds = np.zeros((2, 4, 2))
        data = np.zeros((3, 4))

        msg = (
            "The provided data is not of compatible shape with the "
            r"provided source bounds. \('-', 3, 4\) != \(2, 4\)"
        )
        with self.assertRaisesRegex(ValueError, msg):
            bounded_vinterp.interpolate_conservative(
                target_bounds, source_bounds, data, axis=0
            )

    def test_unexpected_bounds_shape(self):
        # Expecting bounds of size 2 (that is the upper and lower).
        # The source levels should reflect the shape of the data.
        source_bounds = np.zeros((3, 4, 4))
        target_bounds = np.zeros((4, 4, 4))
        data = np.zeros((3, 4))

        msg = r"Unexpected source and target bounds shape. shape\[-1\] != 2"
        with self.assertRaisesRegex(ValueError, msg):
            bounded_vinterp.interpolate_conservative(
                target_bounds, source_bounds, data, axis=0
            )

    def test_not_conservative(self):
        # Where the target does not cover the full extent of the source.
        # |-|-|-|-|-|-|  - Source
        #   |-|-|-|-|    - Target
        def gen_bounds(start, stop, step):
            bounds = np.vstack(
                [
                    np.arange(start, stop - step, step),
                    np.arange(start + step, stop, step),
                ]
            )
            bounds = bounds.transpose((1, 0))
            return bounds.copy()

        source_bounds = gen_bounds(0, 7, 1)
        target_bounds = gen_bounds(1, 6, 1)
        data = np.ones((4, 6))

        msg = "Weights calculation yields a less than conservative result."
        with self.assertRaisesRegex(ValueError, msg):
            bounded_vinterp.interpolate_conservative(
                target_bounds, source_bounds, data, axis=1
            )


if __name__ == "__main__":
    unittest.main()
