import os

import pandas as pd
from pytest import approx, raises

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
