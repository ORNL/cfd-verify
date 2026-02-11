from pathlib import Path

import pandas as pd
import pytest

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


## Regression data ############################################################
@pytest.fixture(scope="package")
def roy_2003():
    """Values from Roy 2003 extracted digitally"""
    hs = [1, 2, 4, 8, 16, 32, 64, 128]
    # Surface pressure at x/R=0.83, Fig. 6
    p8 = [1874.4053654581828,
          1879.7215083533931,
          1885.0376512486057,
          1898.3959687363088,
          1974.5228809387208,
          2193.893485752481,
          2730.3208852926946,
          3931.4490677801496]
    # Surface pressure at x/R=0.23, Fig. 8
    p2 = [12979.06819716408,
          12971.775827143822,
          12962.052667116814,
          12945.847400405131,
          12911.006076975016,
          12914.247130317352,
          13302.36326806212,
          13859.824442943956]
    # Forebody drag, Fig. 10
    ds = [0.98892655,
          0.98830508,
          0.98734463,
          0.98553672,
          0.98293785,
          0.98062147,
          1.00757062,
          1.04683616]
    # Exact values, from text
    exact = [1870, 12980, 0.99]

    data = pd.DataFrame({"hs": hs,
                         "Pressure 0.83": p8,
                         "Pressure 0.23": p2,
                         "Drag": ds})

    return data, exact