"""System and dependency version information capture."""

import sys
from importlib.metadata import PackageNotFoundError, version

from quantumnematode.brain.arch.dtypes import DeviceType


def get_python_version() -> str:
    """Get Python version string.

    Returns
    -------
    str
        Python version (e.g., "3.12.0").
    """
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_package_version(package_name: str) -> str | None:
    """Get installed package version.

    Parameters
    ----------
    package_name : str
        Name of the package.

    Returns
    -------
    str | None
        Package version or None if not installed.
    """
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def get_torch_version() -> str | None:
    """Get PyTorch version if installed.

    Returns
    -------
    str | None
        PyTorch version or None if not installed.
    """
    return get_package_version("torch")


def get_device_type_string(device: DeviceType) -> str:
    """Convert DeviceType enum to string.

    Parameters
    ----------
    device : DeviceType
        Device type enum.

    Returns
    -------
    str
        Device type as string ("cpu", "gpu").
    """
    return device.value


def capture_system_info(
    device_type: DeviceType,
) -> dict[str, str | None]:
    """Capture system and dependency information.

    Parameters
    ----------
    device_type : DeviceType
        Device type used for simulation.

    Returns
    -------
    dict[str, str | None]
        Dictionary with system metadata. Keys python_version and device_type
        are guaranteed to be str. torch_version can be None.
    """
    return {
        "python_version": get_python_version(),
        "torch_version": get_torch_version(),
        "device_type": get_device_type_string(device_type),
    }
