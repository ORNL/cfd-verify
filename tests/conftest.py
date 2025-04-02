from pathlib import Path
import pytest
import pandas as pd
from cfdverify.discretization import CustomDiscretizationError

@pytest.fixture(scope="package")
def hs() -> list:
    return [0.1, 0.2, 0.5]

@pytest.fixture(scope="package")
def fs() -> list:
    return [9.97, 9.88, 9.25]

@pytest.fixture(scope="package")
def gs() -> list:
    return [10.3, 10.6, 11.5]

@pytest.fixture(scope="package")
def dataframe(hs, fs, gs) -> pd.DataFrame:
    return pd.DataFrame({"hs": hs, "fs": fs, "gs": gs})

@pytest.fixture(scope="package")
def osc_dataframe(hs) -> pd.DataFrame:
    osc_data = {"hs": hs,
                "fs": [10.2, 9.5, 10.3],
                "gs": [9.4, 10.4, 10.2]}
    return pd.DataFrame(osc_data)

@pytest.fixture(scope="package")
def least_squared_error_1() -> pd.DataFrame:
    path = Path(__file__).parent.resolve()
    return pd.read_csv(Path(path, "resources", "lse1.csv"))

@pytest.fixture(scope="package")
def custom(dataframe):
    return CustomDiscretizationError(dataframe)