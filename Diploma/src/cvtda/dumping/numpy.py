import os
import typing

import numpy

import cvtda.logging

from .base import BaseDumper


class NumpyDumper(BaseDumper[numpy.ndarray]):
    """
    A dumper for storing numpy arrays on the disk.
    """

    def __init__(self, directory: str):
        self.directory_ = directory

    def get_file_name_(self, name: str):
        return os.path.join(self.directory_, f"{name}.npy")

    def execute(
        self, function: typing.Callable[[typing.Any], numpy.ndarray], name: str, *function_args
    ) -> numpy.ndarray:
        if self.has_dump(name):
            return self.get_dump(name)
        result = function(*function_args)
        self.save_dump(result, name)
        return result

    def save_dump(self, data: numpy.ndarray, name: str):
        file = self.get_file_name_(name)
        cvtda.logging.logger().print(f"Saving the result to {file}")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        numpy.save(file, data)

    def has_dump(self, name: str) -> bool:
        return os.path.exists(self.get_file_name_(name))

    def get_dump_impl_(self, name: str) -> numpy.ndarray:
        file = self.get_file_name_(name)
        cvtda.logging.logger().print(f"Got the result from {file}")
        return numpy.load(self.get_file_name_(name))
