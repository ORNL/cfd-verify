import numpy as np
import pandas as pd
from pytest import approx, raises

import cfdverify.discretization as dis


# Test Discretization Error models
def test_singlepower(dataframe):
    """Test SinglePower class"""
    model = dis.CustomDiscretizationError(dataframe, model=dis.SinglePower)
    test_data = pd.DataFrame({"fs": [10.0, -3.0, 2.0],
                              "gs": [10.0, 3.0, 1.0]},
                             index=["f_est", "alpha", "p"])
    assert model.model.parameter_keys == list(test_data.index)
    pd.testing.assert_frame_equal(model.model.parameters, test_data,
                                  check_dtype=False)
    pd.testing.assert_series_equal(model.f_est, test_data.loc["f_est"],
                                   check_dtype=False,
                                   check_index=False)
    pd.testing.assert_series_equal(model.order, test_data.loc["p"],
                                   check_dtype=False,
                                   check_index=False)
    assert model.model("fs", 0) == approx(10)
    assert model.model("fs", np.array([0, 0.5])) == approx([10, 9.25])

def test_first_and_second_order(dataframe):
    """Test FirstAndSecondOrder class"""
    model = dis.CustomDiscretizationError(dataframe, model=dis.FirstAndSecondOrder)
    test_data = pd.DataFrame({"fs": [10.0, 0.0, -3.0],
                              "gs": [10.0, 3.0, 0.0]},
                             index=["f_est", "alpha_1", "alpha_2"])
    assert model.model.parameter_keys == list(test_data.index)
    pd.testing.assert_frame_equal(model.model.parameters, test_data,
                                  check_dtype=False, atol=1e-6)
    pd.testing.assert_series_equal(model.f_est, test_data.loc["f_est"],
                                   check_dtype=False,
                                   check_index=False)
    pd.testing.assert_series_equal(model.order, pd.Series([1,2]),
                                   check_dtype=False,
                                   check_index=False)
    assert model.model("fs", 0) == approx(10)
    assert model.model("fs", np.array([0, 0.5])) == approx([10, 9.25])

def test_averagevalue(dataframe):
    """Test AverageValue class"""
    model = dis.CustomDiscretizationError(dataframe, model=dis.AverageValue)
    test_data = pd.DataFrame({"fs": [9.7, 0.39230090491866104, 0],
                              "gs": [10.8, 0.6244997998398396, 0]},
                             index=["mean", "std", "order"])
    assert model.model.parameter_keys == list(test_data.index)
    pd.testing.assert_frame_equal(model.model.parameters, test_data,
                                  check_dtype=False)
    pd.testing.assert_series_equal(model.f_est, test_data.loc["mean"],
                                   check_dtype=False,
                                   check_index=False)
    pd.testing.assert_series_equal(model.order, test_data.loc["order"],
                                   check_dtype=False,
                                   check_index=False)
    assert model.model("fs", 0) == approx(9.7)
    assert model.model("fs", np.array([0, 0.5])) == approx([9.7, 9.7])

def test_finestvalue(dataframe):
    """Test FinestValue class"""
    model = dis.CustomDiscretizationError(dataframe, model=dis.FinestValue)
    test_data = pd.DataFrame({"fs": [9.97, 0], "gs": [10.3, 0]},
                             index=["f_est", "order"])
    assert model.model.parameter_keys == list(test_data.index)
    pd.testing.assert_frame_equal(model.model.parameters, test_data,
                                  check_dtype=False)
    pd.testing.assert_series_equal(model.f_est, test_data.loc["f_est"],
                                   check_dtype=False,
                                   check_index=False)
    pd.testing.assert_series_equal(model.order, test_data.loc["order"],
                                   check_dtype=False,
                                   check_index=False)
    assert model.model("fs", 0) == approx(9.97)
    assert model.model("fs", np.array([0, 0.5])) == approx([9.97, 9.97])

def test_maximumvalue(dataframe):
    """Test MaximumValue class"""
    model = dis.CustomDiscretizationError(dataframe, model=dis.MaximumValue)
    test_data = pd.DataFrame({"fs": [9.97, 0], "gs": [11.5, 0]},
                             index=["f_est", "order"])
    assert model.model.parameter_keys == list(test_data.index)
    pd.testing.assert_frame_equal(model.model.parameters, test_data,
                                  check_dtype=False)
    pd.testing.assert_series_equal(model.f_est, test_data.loc["f_est"],
                                   check_dtype=False,
                                   check_index=False)
    pd.testing.assert_series_equal(model.order, test_data.loc["order"],
                                   check_dtype=False,
                                   check_index=False)
    assert model.model("fs", 0) == approx(9.97)
    assert model.model("fs", np.array([0, 0.5])) == approx([9.97, 9.97])

def test_minimumvalue(dataframe):
    """Test MinimumValue class"""
    model = dis.CustomDiscretizationError(dataframe, model=dis.MinimumValue)
    test_data = pd.DataFrame({"fs": [9.25, 0], "gs": [10.3, 0]},
                             index=["f_est", "order"])
    assert model.model.parameter_keys == list(test_data.index)
    pd.testing.assert_frame_equal(model.model.parameters, test_data,
                                  check_dtype=False)
    pd.testing.assert_series_equal(model.f_est, test_data.loc["f_est"],
                                   check_dtype=False,
                                   check_index=False)
    pd.testing.assert_series_equal(model.order, test_data.loc["order"],
                                   check_dtype=False,
                                   check_index=False)
    assert model.model("fs", 0) == approx(9.25)
    assert model.model("fs", np.array([0, 0.5])) == approx([9.25, 9.25])

def test_eçahoekstra2014model(dataframe4):
    """Test Eca2014Model class"""
    model = dis.CustomDiscretizationError(dataframe4, model=dis.EçaHoekstra2014Model)
    test_data = pd.DataFrame({"fs": ["model_p", 10.0, -3.0, 2, np.nan, np.nan],
                              "gs": ["model_p", 10.0, 3.0, 1.0, np.nan, np.nan]},
                             index=["model", "f_est", "alpha_1", "p_1", "alpha_2", "p_2"])
    assert model.model.parameter_keys == list(test_data.index)
    pd.testing.assert_frame_equal(model.model.parameters, test_data,
                                  check_dtype=False)
    pd.testing.assert_series_equal(model.f_est, test_data.loc["f_est"],
                                   check_dtype=False,
                                   check_index=False)
    pd.testing.assert_frame_equal(model.order, test_data.loc[["p_1", "p_2"]],
                                  check_dtype=False)
    assert model.model("fs", 0) == approx(10)
    assert model.model("fs", np.array([0, 0.5])) == approx([10, 9.25])

def test_eçahoekstra2014model_too_few_meshes(dataframe):
    """EçaHoekstra2014Model raises error for fewer than four meshes"""
    with raises(ValueError):
        dis.CustomDiscretizationError(dataframe, model=dis.EçaHoekstra2014Model)

def test_eçahoekstra2014model_order_exceeds_formal_model_1():
    """Test EçaHoekstra2014Model falls back to a 1st order fit when the
    unconstrained fit exceeds the formal order"""
    hs = [0.1, 0.2, 0.35, 0.5]
    fs = [10.0, 9.95, 10.05, 10.0]
    model = dis.CustomDiscretizationError(pd.DataFrame({"hs": hs, "fs": fs}),
                                          model=dis.EçaHoekstra2014Model)
    assert model.model.p_fit == approx(3.2148572417148067)
    test_data = pd.DataFrame({"fs": ["model_1", 9.991967476287877,
                                     0.013090019993698318, 1, np.nan, np.nan]},
                             index=["model", "f_est", "alpha_1", "p_1",
                                    "alpha_2", "p_2"])
    pd.testing.assert_frame_equal(model.model.parameters, test_data,
                                  check_dtype=False)
    assert model.f_est["fs"] == approx(9.991967476287877)
    assert model.order["fs"]["p_1"] == 1
    assert model.model("fs", 0) == approx(9.991967476287877)

def test_eçahoekstra2014model_order_exceeds_formal_model_2():
    """Test EçaHoekstra2014Model falls back to a 2nd order fit when the
    unconstrained fit exceeds the formal order"""
    hs = [0.1, 0.2, 0.35, 0.5]
    fs = [10 - 5*h**4 for h in hs]
    model = dis.CustomDiscretizationError(pd.DataFrame({"hs": hs, "fs": fs}),
                                          model=dis.EçaHoekstra2014Model)
    assert model.model.p_fit == approx(4)
    test_data = pd.DataFrame({"fs": ["model_2", 10.016485557051274,
                                     -1.0934283187393146, 2, np.nan, np.nan]},
                             index=["model", "f_est", "alpha_1", "p_1",
                                    "alpha_2", "p_2"])
    pd.testing.assert_frame_equal(model.model.parameters, test_data,
                                  check_dtype=False)
    assert model.f_est["fs"] == approx(10.016485557051274)
    assert model.order["fs"]["p_1"] == 2
    assert model.model("fs", 0) == approx(10.016485557051274)

def test_eçahoekstra2014model_order_below_half():
    """Test EçaHoekstra2014Model falls back to a mixed 1st/2nd order fit when
    the unconstrained fit is below 0.5"""
    hs = [0.1, 0.2, 0.35, 0.5]
    fs = [10.0, 10.02, 10.03, 10.05]
    model = dis.CustomDiscretizationError(pd.DataFrame({"hs": hs, "fs": fs}),
                                          model=dis.EçaHoekstra2014Model)
    assert model.model.p_fit == approx(0.0834055721198983)
    test_data = pd.DataFrame({"fs": ["model_1and2", 9.979217215507823,
                                     0.22986008568943456, 1,
                                     -0.19220829137216894, 2]},
                             index=["model", "f_est", "alpha_1", "p_1",
                                    "alpha_2", "p_2"])
    pd.testing.assert_frame_equal(model.model.parameters, test_data,
                                  check_dtype=False)
    assert model.f_est["fs"] == approx(9.979217215507823)
    pd.testing.assert_series_equal(model.order["fs"], test_data["fs"].loc[["p_1", "p_2"]],
                                   check_dtype=False, check_names=False)
    assert model.model("fs", 0) == approx(9.979217215507823)
