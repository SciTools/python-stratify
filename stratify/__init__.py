from __future__ import absolute_import

from ._vinterp import (interpolate, interp_schemes,  # noqa: F401
                       extrap_schemes, INTERPOLATE_LINEAR, INTERPOLATE_NEAREST,
                       EXTRAPOLATE_NAN, EXTRAPOLATE_NEAREST,
                       EXTRAPOLATE_LINEAR, PyFuncExtrapolator,
                       PyFuncInterpolator)
from ._bounded_vinterp import (interpolate as  # noqa: F401
    interpolate_conservative)


__version__ = '0.1a3.dev0'
