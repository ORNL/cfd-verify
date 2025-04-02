import os
from pytest import approx, raises
import numpy as np
import pandas as pd
import cfdverify.discretization as dis

## Test DiscretizationError class
# Test constructor options
def test_list_creation(hs, fs):
    model = dis.CustomDiscretizationError(hs, fs)
    pd.testing.assert_series_equal(model.hs, pd.Series(hs, name="hs"))
    pd.testing.assert_frame_equal(model.data,
                                  pd.DataFrame({"System Response Quantity": fs}))

def test_tuple_creation(hs, fs):
    model = dis.CustomDiscretizationError(tuple(hs), tuple(fs))
    pd.testing.assert_series_equal(model.hs, pd.Series(hs, name="hs"))
    pd.testing.assert_frame_equal(model.data,
                                  pd.DataFrame({"System Response Quantity": fs}))

def test_list_tuple_creation(hs, fs):
    model = dis.CustomDiscretizationError(hs, tuple(fs))
    pd.testing.assert_series_equal(model.hs, pd.Series(hs, name="hs"))
    pd.testing.assert_frame_equal(model.data,
                                  pd.DataFrame({"System Response Quantity": fs}))

def test_list_dict_creation(hs, fs):
    key = "fs"
    model = dis.CustomDiscretizationError(hs, {key: fs})
    pd.testing.assert_series_equal(model.hs, pd.Series(hs, name="hs"))
    pd.testing.assert_frame_equal(model.data, pd.DataFrame({key: fs}))

def test_list_invalid_creation(hs):
    with raises(TypeError):
        dis.CustomDiscretizationError(hs, "string")

def test_dict_creation(hs, fs):
    """Test a valid discretization error object can be created with only a dictionary parameter when mesh values are specified by a default value or specified key"""
    key = "fs"
    model = dis.CustomDiscretizationError({"hs": hs, key: fs})
    pd.testing.assert_series_equal(model.hs, pd.Series(hs, name="hs"))
    pd.testing.assert_frame_equal(model.data, pd.DataFrame({key: fs}))
    with raises(ValueError):
        dis.CustomDiscretizationError({"h": hs, key: fs})

def test_dict_creation_with_label(hs, fs):
    """Test a valid discretization error object can be created with only a dictionary parameter when mesh values are specified by a default value or specified key"""
    mesh_key = "sizes"
    key = "fs"
    model = dis.CustomDiscretizationError({mesh_key: hs, key: fs}, mesh_key)
    pd.testing.assert_series_equal(model.hs, pd.Series(hs, name=mesh_key))
    pd.testing.assert_frame_equal(model.data, pd.DataFrame({key: fs}))
    with raises(TypeError):
        model = dis.CustomDiscretizationError({mesh_key: hs, key: fs}, 5)
    with raises(ValueError):
        model = dis.CustomDiscretizationError({"a": hs, key: fs}, mesh_key)

def test_dataframe_creation(dataframe, hs, fs, gs):
    """Test a valid discretization error object can be created with only a Pandas dataframe"""
    model = dis.CustomDiscretizationError(dataframe)
    pd.testing.assert_series_equal(model.hs, pd.Series(hs, name="hs"))
    pd.testing.assert_frame_equal(model.data,
                                  pd.DataFrame({"fs": fs, "gs": gs}))

def test_dataframe_with_label(hs, fs):
    mesh_key = "size"
    key = "fs"
    model = dis.CustomDiscretizationError(pd.DataFrame({mesh_key: hs, key: fs}),
                                          mesh_key)
    pd.testing.assert_series_equal(model.hs, pd.Series(hs, name=mesh_key))
    pd.testing.assert_frame_equal(model.data, pd.DataFrame({key: fs}))
    with raises(TypeError):
        dis.CustomDiscretizationError(pd.DataFrame({mesh_key: hs, key: fs}), 1)
    with raises(ValueError):
        dis.CustomDiscretizationError(pd.DataFrame({mesh_key: hs, key: fs}),
                                      "s")

def test_constructor_exceptions():
    with raises(TypeError):
        dis.CustomDiscretizationError(1)
    with raises(TypeError):
        dis.CustomDiscretizationError("string")

# Test constructor methods
def test_sort(hs, fs):
    """Test _sort method"""
    model = dis.CustomDiscretizationError(list(reversed(hs)),
                                          list(reversed(fs)))
    assert all(model.hs == hs)
    assert all(model.data["System Response Quantity"] == fs)

def test_refinement_ratios(hs, fs):
    """Test _compute_refinement_ratios method"""
    model = dis.CustomDiscretizationError(hs, fs)
    assert model.refinement_ratios == (2.0, 2.5)

def test_u(custom):
    assert custom.u == custom.uncertainty

# Test data methods
def test_len(custom):
    assert len(custom) == 3
    
def test_relative_error(custom):
    """Test relative_error method of DiscretizationError"""
    assert custom.relative_error("fs", 0) == approx(0.09)
    assert custom.relative_error("fs", 2) == approx(0.63)
    pd.testing.assert_frame_equal(custom.relative_error(),
                                  pd.DataFrame({"fs": [0.09, 0.63, 0.63],
                                                "gs": [-0.3, -0.9, -0.9]}))
    pd.testing.assert_series_equal(custom.relative_error("fs"),
                                   pd.Series([0.09, 0.63, 0.63], name="fs"))
        
def test_abs_relative_error(custom):
    """Test abs_relative_error method of DiscretizationError"""
    pd.testing.assert_frame_equal(custom.abs_relative_error(),
                                  pd.DataFrame({"fs": [0.09, 0.63, 0.63],
                                                "gs": [0.3, 0.9, 0.9]}))
    
def test_estimated_error(custom):
    """Test estimated_error method of DiscretizationError"""
    assert custom.estimated_error("fs", 0) == approx(-0.03)
    assert custom.estimated_error("fs", 1) == approx(-0.12)
    assert custom.estimated_error("fs", 2) == approx(-0.75)
    pd.testing.assert_frame_equal(custom.estimated_error(),
                                  pd.DataFrame({"fs": [-0.03, -0.12, -0.75],
                                                "gs": [0.3, 0.6, 1.5]}),
                                  check_dtype=False)
    pd.testing.assert_series_equal(custom.estimated_error("fs"),
                                   pd.Series([-0.03, -0.12, -0.75], name="fs"),
                                   check_dtype=False)

def test_abs_estimated_error(custom):
    """Test abs_estimated_error method of DiscretizationError"""
    pd.testing.assert_frame_equal(custom.abs_estimated_error(),
                                  pd.DataFrame({"fs": [0.03, 0.12, 0.75],
                                                "gs": [0.3, 0.6, 1.5]}),
                                  check_dtype=False)
    
# Test output methods
def test_plot(custom):
    default_name = "DiscretizationError.png"
    custom_name = "Plot.png"
    custom.plot()
    custom.plot("fs")
    custom.plot("gs", 0, custom_name)
    assert os.access(default_name, os.R_OK)
    assert os.access(custom_name, os.R_OK)
    os.remove(default_name)
    os.remove(custom_name)

def test_summarize(custom):
    custom.summarize()
    custom.summarize("fs")

def test_export(custom):
    default_name = "DiscretizationData.csv"
    custom_name = "Data.csv"
    custom.export()
    custom.export(custom_name)
    assert os.access(default_name, os.R_OK)
    assert os.access(custom_name, os.R_OK)
    os.remove(default_name)
    os.remove(custom_name)

## Test concrete DiscretizationError classes
def test_custom(custom):
    assert isinstance(custom.model, dis.SinglePower)
    assert isinstance(custom.error, dis.EstimatedError)
    assert isinstance(custom.uncertainty, dis.GCI)

def test_Classic(dataframe):
    model = dis.Classic(dataframe)
    assert isinstance(model.model, dis.SinglePower)
    assert isinstance(model.error, dis.EstimatedError)
    assert isinstance(model.uncertainty, dis.GCI)

def test_average(dataframe):
    model = dis.Average(dataframe)
    assert isinstance(model.model, dis.AverageValue)
    assert isinstance(model.error, dis.EstimatedError)
    assert isinstance(model.uncertainty, dis.StudentsTDistribution)

# Test Discretization Error models
def test_singlepower(dataframe):
    """Test SinglePower class"""
    model = dis.CustomDiscretizationError(dataframe, model=dis.SinglePower)
    test_data = pd.DataFrame({"fs": [10.0, 2.0, -3.0],
                              "gs": [10.0, 1.0, 3.0]},
                             index=["f_est", "p", "alpha"])
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

# Test uncertainty models
def test_gci(dataframe):
    model = dis.CustomDiscretizationError(dataframe, uncertainty=dis.GCI)
    test_data = pd.DataFrame({"fs": [1.25 * 0.09 / (2**2 - 1),
                                     1.25 * 0.63 / (2.5**2 - 1),
                                     1.25 * 0.63 / (2.5**2 - 1) * 2.5**2],
                              "gs": [1.25 * 0.3 / (2**1 - 1),
                                     1.25 * 0.9 / (2.5**1 - 1),
                                     1.25 * 0.9 / (2.5**1 - 1) * 2.5**1]})
    pd.testing.assert_series_equal(model.uncertainty("fs"), test_data["fs"])
    pd.testing.assert_series_equal(model.uncertainty("gs"), test_data["gs"])
    assert model.uncertainty("fs", 2) == approx(test_data["fs"][2])
    assert model.uncertainty("fs", 2, 2) == approx(test_data["fs"][2] * 2/1.25)
    assert model.uncertainty("fs", 0, normalize=True) == approx(test_data["fs"][0]/9.97)
    assert model.uncertainty("gs", 2, normalize=True) == approx(test_data["gs"][2]/11.5)
    pd.testing.assert_series_equal(model.uncertainty("fs", normalize=True),
                                   test_data["fs"]/np.array([9.97, 9.88, 9.25]))
    
def test_gci_lse1(least_squared_error_1):
    model = dis.CustomDiscretizationError(least_squared_error_1,
                                          uncertainty=dis.GCI)
    assert model.u("C_l", 0) > model.error("C_l", 0)
    pd.testing.assert_series_equal(model.u("C_l"), 1.25*model.error("C_l"))
    
def test_studentstdistribution(osc_dataframe):
    model = dis.CustomDiscretizationError(osc_dataframe,
                                          model=dis.AverageValue,
                                          uncertainty=dis.StudentsTDistribution)
    test_data = pd.DataFrame({"fs": [1.0828105247765283,
                                     1.0828105247765283,
                                     1.0828105247765283],
                              "gs": [1.3144821215951197,
                                     1.3144821215951197,
                                     1.3144821215951197]})
    pd.testing.assert_series_equal(model.uncertainty("fs"), test_data["fs"])
    pd.testing.assert_series_equal(model.uncertainty("gs"), test_data["gs"])
    assert model.uncertainty("fs", 2) == approx(test_data["fs"][2])
    assert model.uncertainty("gs", 0, 0.1) == approx(0.8920703300105804)
    
def test_factorofsafety(dataframe):
    model = dis.CustomDiscretizationError(dataframe,
                                          uncertainty=dis.FactorOfSafety)
    test_data = pd.DataFrame({"fs": [0.09, 0.36, 2.25],
                              "gs": [0.9, 1.8, 4.5]})
    pd.testing.assert_series_equal(model.uncertainty("fs"), test_data["fs"])
    pd.testing.assert_series_equal(model.uncertainty("gs"), test_data["gs"])
    assert model.uncertainty("fs", 2) == approx(test_data["fs"][2])
    assert model.uncertainty("fs", 2, 2) == approx(test_data["fs"][2] * 2/3)