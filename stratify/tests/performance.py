"""
Functions that may be used to measure performance of a component.

"""
import numpy as np
import stratify


def src_data(shape=(400, 500, 100)):
    z = np.tile(np.linspace(0, 100, shape[-1]),
                np.prod(shape[:2])).reshape(shape)
    fz = np.arange(np.prod(shape)).reshape(shape)
    return z, fz 


def interp_and_extrap(shape,
                      interp=stratify.INTERPOLATE_LINEAR,
                      extrap=stratify.EXTRAPOLATE_NEAREST):
    z, fz = src_data(shape)
    stratify.interpolate(np.linspace(-20, 120, 50), z, fz,
                         interpolation=interp, extrapolation=extrap)


if __name__ == '__main__':
    interp_and_extrap(shape=(500, 600, 100))
