import numpy as np
import pandas as pd
from pytest import approx

import cfdverify.discretization as dis


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

def test_eçahoekstra2014uncertainty(dataframe4):
    model = dis.EçaHoekstra2014(dataframe4)
    test_data = 1.25 * pd.DataFrame({"fs": [0.03, 0.12, 0.3675, 0.75],
                                     "gs": [0.3, 0.6, 1.05, 1.5]})
    pd.testing.assert_series_equal(model.uncertainty("fs"), test_data["fs"])
    pd.testing.assert_series_equal(model.uncertainty("gs"), test_data["gs"])
    assert model.uncertainty("fs", 2) == approx(test_data["fs"][2])
