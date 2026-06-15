import typing


def dump_name_concat(dump_name: typing.Optional[str], extra_path: str):
    return None if dump_name is None else f"{dump_name}/{extra_path}"
