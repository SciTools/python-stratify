"""
Functions that may be used to measure performance of a component.

"""
import argparse

import numpy as np
import stratify


def src_data(shape=(400, 500, 100)):
    z = np.tile(np.linspace(0, 100, shape[-1]),
                np.prod(shape[:2])).reshape(shape)
    fz = np.arange(np.prod(shape)).reshape(shape)
    return z, fz 


def interp_and_extrap(shape,
                      interp='linear',
                      extrap='nearest'):
    z, fz = src_data(shape)
    stratify.interpolate(np.linspace(-20, 120, 50), z, fz,
                         interpolation=interp, extrapolation=extrap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an interpolation.')
    parser.add_argument('--shape', default='500,600,100',
                        help='The shape of the array to interpolate. Comma separated (no spaces).')
    parser.add_argument('--interp', default='linear',
                        help='The interpolation scheme to use.')
    parser.add_argument('--extrap', default='nearest',
                        help='The extrapolation scheme to use.')

    args = parser.parse_args()
    interp_and_extrap(shape=[int(length) for length in args.shape.split(',')],
                      interp=args.interp, extrap=args.extrap)
