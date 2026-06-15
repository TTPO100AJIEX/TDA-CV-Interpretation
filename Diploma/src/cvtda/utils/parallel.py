import joblib
import typing
import inspect

import cvtda.logging
import cvtda.dumping


def parallel(
    function,
    iterable,
    n_jobs: int = -1,
    return_as: str = "list",
    num_parameters: typing.Optional[str] = None,
):
    """
    Utility wrapper around :obj:`joblib.Parallel` that correctly transfers
    :mod:`cvtda.logging` and :mod:`cvtda.dumping` contexts to the created threads.

    Parameters
    ----------
    function : ``Callable``
        The function to execute.
    iterable : ``Iterable``
        The set of elements to execute the function with. Each element will be passed to `function`.
    n_jobs : ``int``, default: ``-1``
        The number of jobs to use for the computation. See :mod:`joblib` for details.
    return_as : ``str``, default: ``list``
        The return value type of :obj:`joblib.Parallel`. See :obj:`joblib.Parallel` for details.

    Returns
    -------
    ``joblib.Parallel``
        The parallel instance performing the computations.
    """

    def init_worker(logger, dumper):
        cvtda.logging.BaseLogger.current_logger = logger
        cvtda.dumping.BaseDumper.current_dumper = dumper

    initargs = (cvtda.logging.logger(), cvtda.dumping.dumper())
    num_parameters = num_parameters or len(inspect.signature(function).parameters)
    with joblib.parallel_backend(backend="loky", initializer=init_worker, initargs=initargs):
        if num_parameters == 1:
            return joblib.Parallel(return_as=return_as, n_jobs=n_jobs)(
                joblib.delayed(function)(item) for item in iterable
            )
        else:
            return joblib.Parallel(return_as=return_as, n_jobs=n_jobs)(
                joblib.delayed(function)(*item) for item in iterable
            )
