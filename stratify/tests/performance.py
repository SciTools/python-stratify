"""
Functions that may be used to measure performance of a component.

"""
import dask.array as da
import numpy as np

import stratify


def src_data(shape=(400, 500, 100), lazy=False):
    z = np.tile(np.linspace(0, 100, shape[-1]), np.prod(shape[:2])).reshape(shape)
    if lazy:
        fz = da.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    else:
        fz = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    return z, fz


def interp_and_extrap(
    shape,
    lazy,
    interp=stratify.INTERPOLATE_LINEAR,
    extrap=stratify.EXTRAPOLATE_NEAREST,
):
    z, fz = src_data(shape, lazy)
    tgt = np.linspace(-20, 120, 50)
    result = stratify.interpolate(
        tgt,
        z,
        fz,
        interpolation=interp,
        extrapolation=extrap,
    )
    if isinstance(result, da.Array):
        print("lazy calculation")
        print(result.chunks)
        result.compute()
    else:
        print("non-lazy calculation")


if __name__ == "__main__":
    import sys

    lazy = "lazy" in sys.argv[1:]
    interp_and_extrap(shape=(500, 600, 100), lazy=lazy)
