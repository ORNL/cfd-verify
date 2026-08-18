import pandas as pd
from pytest import approx

import cfdverify.discretization as dis


# Test error models
def test_estimatederror(dataframe):
    """Test EstimatedError class"""
    model = dis.CustomDiscretizationError(dataframe, error=dis.EstimatedError)
    test_data = pd.DataFrame({"fs": [-0.03, -0.12, -0.75],
                              "gs": [0.3, 0.6, 1.5]})
    pd.testing.assert_frame_equal(model.error.get_data(None),
                                  dataframe[["fs", "gs"]])
    pd.testing.assert_series_equal(model.error.get_data("fs"), dataframe["fs"])
    # Returns element-wise computation of data, which has generic object type
    pd.testing.assert_frame_equal(model.error(), test_data, check_dtype=False)
    pd.testing.assert_series_equal(model.error("fs"), test_data["fs"])
    assert model.error("fs", 0) == approx(-0.03)

def test_relativeerror(dataframe):
    """Test RelativeError class"""
    model = dis.CustomDiscretizationError(dataframe, error=dis.RelativeError)
    test_data = pd.DataFrame({"fs": [0.09, 0.63, 0.63],
                              "gs": [-0.3, -0.9, -0.9]})
    pd.testing.assert_frame_equal(model.error.get_data(None),
                                  dataframe[["fs", "gs"]])
    pd.testing.assert_series_equal(model.error.get_data("fs"), dataframe["fs"])
    # Returns element-wise computation of data, which has generic object type
    pd.testing.assert_frame_equal(model.error(), test_data, check_dtype=False)
    pd.testing.assert_series_equal(model.error("fs"), test_data["fs"])
    assert model.error("fs", 0) == approx(0.09)
