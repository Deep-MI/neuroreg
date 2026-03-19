# from nireg._version import __version__  # noqa: F401
from nireg.register import register, register_pyramid, register_pyramid_sym, register_surface  # noqa: F401
from nireg.transforms import (  # noqa: F401
    LINEAR_RAS_TO_RAS,
    LINEAR_VOX_TO_VOX,
    LTA,
    convert_transform_type,
)
from nireg.utils._config import sys_info  # noqa: F401
