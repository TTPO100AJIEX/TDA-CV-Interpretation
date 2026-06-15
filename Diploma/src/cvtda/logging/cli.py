import typing

import tqdm
import tqdm.contrib

from .base import BaseLogger

T = typing.TypeVar("T")


class CLILogger(BaseLogger):
    """
    A logger that prints to the standard output.
    """

    def __init__(self):
        pass

    def verbosity(self) -> int:
        return 2

    def print(self, data: T, *args) -> None:
        print(data, *args)

    def pbar(
        self, data: typing.Iterable[T], total: int = None, desc: typing.Optional[str] = None
    ) -> typing.Iterable[T]:
        return tqdm.tqdm(data, total=total, desc=desc)

    def zip(self, *iterables, desc: typing.Optional[str] = None):
        return tqdm.contrib.tzip(*iterables, desc=desc)

    def set_pbar_postfix(self, pbar: tqdm.tqdm, data: dict):
        pbar.set_postfix(data)
