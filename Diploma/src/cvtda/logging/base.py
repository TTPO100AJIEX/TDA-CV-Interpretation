import abc
import typing

T = typing.TypeVar("T")


class BaseLogger(abc.ABC):
    """
    Base class for loggers with context management support.
    """

    current_logger = None

    def __enter__(self):
        self.__previous = BaseLogger.current_logger
        BaseLogger.current_logger = self
        return self

    def __exit__(self, *args):
        BaseLogger.current_logger = self.__previous

    @abc.abstractmethod
    def verbosity(self) -> int:
        """
        Gets the verbosity level of the logger.

        Returns
        -------
        ``int``
            The verbosity level.
        """
        pass

    @abc.abstractmethod
    def print(self, data: T, *args) -> None:
        """
        Prints data and args using the logger.

        Parameters
        ----------
        data : ``Any``
            The data to print.
        args : ``Any``
            Extra arguments to use for printing.
        """
        pass

    @abc.abstractmethod
    def pbar(
        self, data: typing.Iterable[T], total: typing.Optional[int] = None, desc: typing.Optional[str] = None
    ) -> typing.Iterable[T]:
        """
        Wraps an iterable in a progress bar that acts exactly
        like the original iterable, but prints a dynamically updating
        progressbar every time a value is requested.

        Parameters
        ----------
        data : ``Iterable``
            The original iterable.
        total : ``int``, optional
            The total number of elements in the iterable if len() is not available.
        desc : ``str``, optional
            The description to print with the progress bar.
        """
        pass

    @abc.abstractmethod
    def zip(self, *iterables, desc: typing.Optional[str] = None):
        """
        Equivalent of builtin `zip` with a progress bar printed by the logger.

        Parameters
        ----------
        iterables : ``list[Iterable]``
            The iterables to zip.
        desc : ``str``, optional
            The description to print with the progress bar.
        """
        pass

    @abc.abstractmethod
    def set_pbar_postfix(self, pbar: typing.Any, data: dict):
        """
        Sets a progress bar postfix (additional stats) with automatic formatting based on datatype.

        Parameters
        ----------
        pbar : ``Any``
            The progress bar instance.
        data : ``dict``
            The data to print with the progress bar.
        """
        pass
