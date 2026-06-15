import abc
import typing

T = typing.TypeVar("T")


class BaseDumper(abc.ABC, typing.Generic[T]):
    """
    Base class for dumping with context management support.
    """

    current_dumper = None

    def __enter__(self):
        self.__previous = BaseDumper.current_dumper
        BaseDumper.current_dumper = self
        return self

    def __exit__(self, *args):
        BaseDumper.current_dumper = self.__previous

    @abc.abstractmethod
    def execute(self, function: typing.Callable[[typing.Any], T], name: str, *function_args) -> T:
        """
        Gets the result from dump, if available, or executes the function and dumps the output.

        Parameters
        ----------
        function : ``Callable``
            The function to execute.
        name : ``str``
            The name of the dump.
        function_args : ``list[Any]``
            Extra arguments to pass to the function.

        Returns
        -------
        ``Any``
            The output of the function.
        """
        pass

    @abc.abstractmethod
    def save_dump(self, data: T, name: str):
        """
        Dumps data with the given name.

        Parameters
        ----------
        data : ``Any``
            The data to dump.
        name : ``str``
            The name of the dump.
        """
        pass

    @abc.abstractmethod
    def has_dump(self, name: str) -> bool:
        """
        Checks if the dump with a given name is available.

        Parameters
        ----------
        name : ``str``
            The name of the dump.

        Returns
        -------
        ``bool``
            True, if the dump is available. False otherwise.
        """
        pass

    def get_dump(self, name: str) -> T:
        """
        Ensures the dump with a given name is available and gets it.
        Throws :exc:`AssertionError` if the dump is not available.

        Parameters
        ----------
        name : ``str``
            The name of the dump.

        Returns
        -------
        ``Any``
            The contents of the dump.
        """
        assert self.has_dump(name), f"There is no dump at {name}"
        return self.get_dump_impl_(name)

    @abc.abstractmethod
    def get_dump_impl_(self, name: str) -> T:
        """
        Gets the dump with a given name without checking if it is available.

        Parameters
        ----------
        name : ``str``
            The name of the dump.

        Returns
        -------
        ``Any``
            The contents of the dump.
        """
        pass
