import typing

from .base import BaseDumper

T = typing.TypeVar("T")


class DevNullDumper(BaseDumper[T]):
    """
    A dumper that actually does not save anything.
    """

    def execute(self, function: typing.Callable[[typing.Any], T], name: str, *function_args) -> T:
        return function(*function_args)

    def save_dump(self, data: T, name: str):
        return

    def has_dump(self, name: str) -> bool:
        return False

    def get_dump_impl_(self, name: str) -> T:
        raise NotImplementedError
