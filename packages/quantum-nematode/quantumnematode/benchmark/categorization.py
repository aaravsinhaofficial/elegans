"""Benchmark categorization logic."""

from quantumnematode.experiment.metadata import ExperimentMetadata


def get_environment_category(
    grid_size: int,
    *,
    predators_enabled: bool = False,
) -> str:
    """Get environment category string.

    Parameters
    ----------
    grid_size : int
        Grid size.
    predators_enabled : bool, optional
        Whether predators are enabled, by default False.

    Returns
    -------
    str
        Environment category (e.g., "foraging_small", "predator_small").
    """
    size_category = "small" if grid_size <= 20 else "medium" if grid_size <= 50 else "large"

    if predators_enabled:
        return f"predator_{size_category}"
    return f"foraging_{size_category}"


def determine_benchmark_category(metadata: ExperimentMetadata) -> str:
    """Determine benchmark category from experiment metadata.

    Parameters
    ----------
    metadata : ExperimentMetadata
        Experiment metadata.

    Returns
    -------
    str
        Benchmark category string (e.g., "foraging_medium", "predator_small").
    """
    return get_environment_category(
        metadata.environment.grid_size,
        predators_enabled=metadata.environment.predators_enabled,
    )


def get_category_directory(category: str) -> str:
    """Get benchmark storage directory path for a category.

    Parameters
    ----------
    category : str
        Benchmark category.

    Returns
    -------
    str
        Relative directory path (e.g., "foraging_medium").
    """
    return category
