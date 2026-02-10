import pytest
import numpy as np
import pandas as pd
import cfdverify.code as code

@pytest.fixture(scope="module")
def base_index() -> pd.Index:
    return pd.Index([1, 0.5, 0.25, 0.125, 0.0625], name='\u0394')

@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    data = pd.DataFrame({"1st": [1/2**4, 1, 1/2, 1/2**2, 1/2**3],
                        "2nd": [1/4**4, 1, 1/4, 1/4**2, 1/4**3],
                        "3rd": [1/8**4, 1, 1/8, 1/8**2, 1/8**3],
                        "2nd_noise": [0.99/4**4, 1, 1.05/4, 1.01/4**2, 0.98/4**3],
                        "2nd_low": [1.04/4**4, 1, 1.01/4, 1.02/4**2, 1.03/4**3],
                        },
                        index=[0.0625, 1, 0.5, 0.25, 0.125])
    return data

@pytest.fixture(scope="module")
def relative_data() -> pd.DataFrame:
    data = pd.DataFrame({"1st": [1, 1/2, 1/2**2, 1/2**3, 1/2**8],
                            "2nd": [1, 1/4, 1/4**2, 1/4**3, 1/4**8],
                            "3rd": [1, 1/8, 1/8**2, 1/8**3, 1/8**8],
                            "2nd_noise": [1, 1.05/4, 1.01/4**2, 0.98/4**3, 0.99/4**8],
                            "2nd_low": [1, 1.01/4, 1.02/4**2, 1.03/4**3, 1.04/4**8],
                            },
                            index=pd.Index([1, 0.5, 0.25, 0.125, 0.00390625], name='\u0394'))
    return data

@pytest.fixture(scope="module")
def base_model(data) -> code.OrderOfAccuracy:
    return code.OrderOfAccuracy(data)

## Test OrderOfAccuracy class
# Test constructor
def test_dataframe_creation(base_model, base_index):
    data = pd.DataFrame({"1st": [1, 1/2, 1/2**2, 1/2**3, 1/2**4],
                        "2nd": [1, 1/4, 1/4**2, 1/4**3, 1/4**4],
                        "3rd": [1, 1/8, 1/8**2, 1/8**3, 1/8**4],
                        "2nd_noise": [1, 1.05/4, 1.01/4**2, 0.98/4**3, 0.99/4**4],
                        "2nd_low": [1, 1.01/4, 1.02/4**2, 1.03/4**3, 1.04/4**4],
                        },
                        index=base_index)
    pd.testing.assert_frame_equal(base_model.data, data)

def test_dataframe_relative_error(relative_data):
    error_model = code.OrderOfAccuracy(relative_data, True)
    data = pd.DataFrame({"1st": [1, 1/2, 1/2**2, 1/2**3],
                        "2nd": [1, 1/4, 1/4**2, 1/4**3],
                        "3rd": [1, 1/8, 1/8**2, 1/8**3],
                        "2nd_noise": [1, 1.05/4, 1.01/4**2, 0.98/4**3],
                        "2nd_low": [1, 1.01/4, 1.02/4**2, 1.03/4**3],
                        },
                        index=pd.Index([1, 0.5, 0.25, 0.125], name='\u0394'))
    ref = pd.Series([1/2**8, 1/4**8, 1/8**8, 0.99/4**8, 1.04/4**8],
                    pd.Index(["1st", "2nd", "3rd", "2nd_noise", "2nd_low"]),
                    name=0.00390625)
    orders = pd.Series([1, 2, 3, 2.003624892423779, 1.9858541179084082],
                       pd.Index(["1st Order", "2nd Order", "3rd Order", "2nd_noise Order", "2nd_low Order"]))
    pd.testing.assert_series_equal(error_model.data["1st"], data["1st"] - 0.00390625)
    pd.testing.assert_series_equal(error_model.reference_result, ref)
    pd.testing.assert_series_equal(error_model.get_average_orders(), orders, rtol=0.02)

# Test constructor methods
def test_compute_refinement_ratios(base_model, base_index):
    r = pd.Series([np.nan, 2, 2, 2, 2], base_index, name="Refinement Ratio")
    pd.testing.assert_series_equal(base_model.r, r)

def test_observed_order_float(base_model):
    assert base_model._observed_order(2, 1, 2) == 1

def test_observed_order_array(base_model):
    coarse = np.array([2, 12])
    fine = np.array([1, 3])
    r = np.array([2, 2])
    order = base_model._observed_order(coarse, fine, r)
    np.testing.assert_equal(order, np.array([1, 2]))

def test_order_key(base_model):
    assert base_model._order_key("test") == "test Order"

# Test orders
def test_1st_order(base_model, base_index):
    first = pd.Series([np.nan, 1, 1, 1, 1], base_index, float, name="1st Order")
    pd.testing.assert_series_equal(base_model.p_hat["1st Order"], first)

def test_2nd_order(base_model, base_index):
    second = pd.Series([np.nan, 2, 2, 2, 2], base_index, float, name="2nd Order")
    pd.testing.assert_series_equal(base_model.p_hat["2nd Order"], second)

def test_3rd_order(base_model, base_index):
    third = pd.Series([np.nan, 3, 3, 3, 3], base_index, float, name="3rd Order")
    pd.testing.assert_series_equal(base_model.p_hat["3rd Order"], third)

# Test generic methods
def test_get_average_orders(base_model, base_index):
    avg = pd.Series([1, 2, 3, 2.003624892423779, 1.9858541179084082], pd.Index(["1st Order", "2nd Order", "3rd Order", "2nd_noise Order", "2nd_low Order"]))
    pd.testing.assert_series_equal(base_model.get_average_orders(), avg)
