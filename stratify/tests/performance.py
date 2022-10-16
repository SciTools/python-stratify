"""
Functions that may be used to measure performance of a component.

"""
import dask.array as da
import numpy as np

import stratify


def src_data(shape=(400, 500, 100)):
    z = np.tile(np.linspace(0, 100, shape[-1]), np.prod(shape[:-1])).reshape(shape)
    fz = np.arange(np.prod(shape)).reshape(shape)
    return z, fz


def interp_and_extrap(
    shape, interp=stratify.INTERPOLATE_LINEAR, extrap=stratify.EXTRAPOLATE_NAN
):
    z, fz = src_data(shape)
    fz_lazy = da.asarray(fz)
    print(fz_lazy.chunks)
    tgt = np.linspace(-20, 120, 50)  # * np.ones(shape[:-1] + (50,))
    # tgt = np.array([2])
    # tgt = da.asarray(tgt, chunks=('auto', 'auto', None))
    # print(tgt)
    # r1 = stratify.interpolate(tgt, z, fz,
    #                     interpolation=interp, extrapolation=extrap)
    r2 = stratify.interpolate(
        tgt, z, fz_lazy, interpolation=interp, extrapolation=extrap
    )
    print(r2.chunks)
    r2.compute()
    # assert (r1 == r2.compute()).all()


if __name__ == "__main__":
    # print(stratify.__file__)
    # print(stratify.__version__)
    # interp_and_extrap(shape=(1872, 64 * 128, 22))
    interp_and_extrap(shape=(2000, 1000, 22))
