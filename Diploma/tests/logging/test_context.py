import cvtda.logging


def is_current_logger(expected_logger_class):
    return isinstance(cvtda.logging.logger(), expected_logger_class)


def test_context():
    assert is_current_logger(cvtda.logging.CLILogger)
    with cvtda.logging.DevNullLogger():
        assert is_current_logger(cvtda.logging.DevNullLogger)
        with cvtda.logging.CLILogger():
            assert is_current_logger(cvtda.logging.CLILogger)
        assert is_current_logger(cvtda.logging.DevNullLogger)
    assert is_current_logger(cvtda.logging.CLILogger)
