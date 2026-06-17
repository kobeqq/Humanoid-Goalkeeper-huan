from typing import Any

class SubTerrain:
    height_field_raw: Any
    def __init__(
        self,
        name: str,
        width: int,
        length: int,
        vertical_scale: float,
        horizontal_scale: float,
    ) -> None: ...

def random_uniform_terrain(
    terrain: SubTerrain,
    min_height: float,
    max_height: float,
    step: int,
    downsampled_scale: float,
) -> None: ...
