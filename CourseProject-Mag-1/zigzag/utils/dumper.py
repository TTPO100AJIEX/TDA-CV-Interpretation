from __future__ import annotations
import os
import shutil
import typing
import pathlib
import inspect

import numpy
import torch
import pandas
import scipy.sparse

import cvtda.logging

from cvtda.dumping import BaseDumper

T = typing.TypeVar("T")

EXTENSIONS = ["pt", "pth", "npy", "csv", "npz", ""]


def get_data_type(data) -> type:
    if type(data) == "type":
        return data
    if type(data) == list:
        return typing.List
    if isinstance(data, torch.nn.Module):
        return torch.nn.Module
    return type(data)


def get_extension(data_type):
    if data_type.__name__ == "List" or data_type == list:
        return ""
    if data_type == torch.Tensor:
        return "pt"
    if data_type == torch.nn.Module:
        return "pth"
    if data_type == numpy.ndarray:
        return "npy"
    if data_type == pandas.DataFrame:
        return "csv"
    if data_type == scipy.sparse.csr_matrix:
        return "npz"
    assert False, f"Unsuppported data type: {data_type}"


class UniversalDumper(BaseDumper[torch.Tensor]):
    def __init__(self, directory: str):
        self.directory_ = directory

    def clear(self):
        shutil.rmtree(self.directory_, ignore_errors=True)

    def make_subdumper(self, subdir: str) -> UniversalDumper:
        return UniversalDumper(f"{self.directory_}/{subdir}")

    def get_file_name_(self, name: str, ext: typing.Optional[str]):
        name = f"{name}.{ext}" if ext != "" else name
        return os.path.join(self.directory_, name)

    def get_existing_file_name_(self, name: str, ext: typing.Optional[str] = None) -> typing.Optional[str]:
        if ext is not None:
            file = self.get_file_name_(name, ext)
            return file if os.path.exists(file) else None
        exist = set([ext for ext in EXTENSIONS if os.path.exists(self.get_file_name_(name, ext))])
        assert len(exist) <= 1, f"Ambiguous dump. Multiple files exist: {exist}"
        return self.get_file_name_(name, next(iter(exist))) if len(exist) == 1 else None

    def execute(self, function: typing.Callable[[typing.Any], T], name: str, *function_args, **function_kwargs) -> T:
        return_type = inspect.signature(function).return_annotation
        if self.has_dump(name, get_extension(return_type)):
            return self.get_dump(name, get_extension(return_type))
        result = function(*function_args, **function_kwargs)
        self.save_dump(result, name)
        return result

    def save_dump(self, data: T, name: str):
        file = self.get_file_name_(name, get_extension(get_data_type(data)))
        cvtda.logging.logger().print(f"Saving the result to {file}")
        os.makedirs(os.path.dirname(file), exist_ok=True)

        data_type = get_data_type(data)
        if data_type.__name__ in ("list", "List"):
            with cvtda.logging.DevNullLogger():
                for i, item in enumerate(data):
                    self.save_dump(item, f"{name}/{i}")
            return
        if data_type == torch.Tensor or data_type == torch.nn.Module:
            return torch.save(data, file)
        if data_type == numpy.ndarray:
            return numpy.save(file, data)
        if data_type == pandas.DataFrame:
            return data.to_csv(file, index=False)
        if data_type == scipy.sparse.csr_matrix:
            return scipy.sparse.save_npz(file, data)
        assert False, f"Unsuppported data type: {type(data)}"

    def has_dump(self, name: str, ext: typing.Optional[str] = None) -> bool:
        return self.get_existing_file_name_(name, ext) is not None

    def get_dump_impl_(self, name: str, ext: typing.Optional[str] = None) -> T:
        file = self.get_existing_file_name_(name, ext)
        cvtda.logging.logger().print(f"Got the result from {file}")
        if file.endswith(name):
            files: typing.List[typing.Tuple[str, str]] = []
            for filename in os.listdir(file):
                path = pathlib.Path(filename)
                try:
                    files.append((int(path.stem), path.suffix[1:]))
                except:
                    pass
            with cvtda.logging.DevNullLogger():
                return [self.get_dump(f"{name}/{file}", ext) for file, ext in sorted(files, key=lambda file: file[0])]
        elif file.endswith(".pt"):
            return torch.load(file)
        elif file.endswith(".pth"):
            return torch.load(file, weights_only=False)
        elif file.endswith(".npy"):
            return numpy.load(file)
        elif file.endswith("csv"):
            return pandas.read_csv(file)
        elif file.endswith(".npz"):
            return scipy.sparse.load_npz(file)
        else:
            assert False, f"Unsuppported dump filename: {file}"
