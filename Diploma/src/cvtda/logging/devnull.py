import typing

from .base import BaseLogger

T = typing.TypeVar("T")


class DevNullLogger(BaseLogger):
    """
    A logger that suppresses all messages.
    """

    def print(self, data: T, *args) -> None:
        pass

    def verbosity(self) -> int:
        return 0

    def pbar(
        self, data: typing.Iterable[T], total: int = None, desc: typing.Optional[str] = None
    ) -> typing.Iterable[T]:
        return data

    def zip(self, *iterables, desc: typing.Optional[str] = None):
        return zip(*iterables)

    def set_pbar_postfix(self, pbar: typing.Any, data: dict):
        pass
