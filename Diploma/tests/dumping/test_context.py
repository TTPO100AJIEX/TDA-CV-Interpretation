import cvtda.dumping


def is_current_dumper(expected_dumper_class):
    return isinstance(cvtda.dumping.dumper(), expected_dumper_class)


def test_context():
    assert is_current_dumper(cvtda.dumping.NumpyDumper)
    with cvtda.dumping.DevNullDumper():
        assert is_current_dumper(cvtda.dumping.DevNullDumper)
        with cvtda.dumping.NumpyDumper("/tmp"):
            assert is_current_dumper(cvtda.dumping.NumpyDumper)
        assert is_current_dumper(cvtda.dumping.DevNullDumper)
    assert is_current_dumper(cvtda.dumping.NumpyDumper)
