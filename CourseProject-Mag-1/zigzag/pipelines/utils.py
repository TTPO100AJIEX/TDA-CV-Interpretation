import enum
import typing
import dataclasses


@dataclasses.dataclass()
class Params:
    k_neighbors: int
    dimension: int
    num_layers: typing.Optional[int] = None


class Verbosity(enum.Enum):
    LOGS = 1
    ONLY_PROGRESSBAR = 2
