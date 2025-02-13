TYPES= ['float32','float64','complex64','complex128']

def pytest_addoption(parser):
    parser.addoption(
        "--sizes",
        action="store",
        default="100,200",
        help="Comma-separated list of matrix sizes."
    )
    parser.addoption(
        "--dtypes",
        action="store",
        default="float32,float64,complex64,complex128",
        help="Comma-separated list of dtypes."
    )
    parser.addoption(
        "--ratios",
        action="store",
        default="0.02,0.015,0.01",
        help="Comma-separated list of ratios of the spectrum to compute when using partial solver."
    )


def pytest_generate_tests(metafunc):
    if "size" in metafunc.fixturenames:
        sizes = metafunc.config.getoption("sizes").split(",")
        sizes = [int(s.strip()) for s in sizes]
        metafunc.parametrize("size", sizes)

    if "dtype" in metafunc.fixturenames:
        dtypes = metafunc.config.getoption("dtypes").split(",")
        dtypes = [dt.strip() for dt in dtypes]
        assert all(dt in TYPES for dt in dtypes), f"Invalid dtypes: {dtypes}, valid dtypes: {TYPES}"
        metafunc.parametrize("dtype", dtypes)

    if "ratio" in metafunc.fixturenames:
        ratios = metafunc.config.getoption("ratios").split(",")
        ratios = [float(rt.strip()) for rt in ratios]
        assert all(rt<1 for rt in ratios), f"Invalid ratios: {ratios}, ratios must be less than 1"
        metafunc.parametrize("ratio", ratios)