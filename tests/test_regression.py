from pytest import approx

import cfdverify.utils as utils
import cfdverify.discretization as dis


def test_asme_procedure_phi1():
    """Test against ASME JFE announcement
    
    This test tests the code against the data contained in the American Society
    of Mechanical Engineer's Journal of Fluids Engineering announcement
    "Procedure for Estimation and Reporting of Uncertainty Due to 
    Discretization in CFD Applications." Specifically, this test compares data 
    for the first response quantity, the dimensionless reattachment length.

    References
    ----------
    2008, Procedure for Estimation and Reporting of Uncertainty Due to 
    Discretization in CFD Applications, Journal of Fluids Engineering, 
    130(7): 078001. https://doi.org/10.1115/1.2960953
    """
    N = [18_000, 8_000, 4_500]
    V = 1 # placeholder volume
    mesh_sizes = utils.mesh_size(V, N, 2)
    phi1 = [6.063, 5.972, 5.863]
    model = dis.Classic(mesh_sizes, phi1)
    assert model.refinement_ratios == approx([1.5, 1.333], abs=0.0005)
    assert model.order["System Response Quantity"] == approx(1.53, abs=0.005)
    assert model.f_est["System Response Quantity"] == approx(6.1685,
                                                             abs=0.00005)
    assert model.uncertainty("System Response Quantity", 0, normalize=True) == approx(0.022, abs=0.0005)


def test_asme_procedure_phi2():
    """Test against ASME JFE announcement
    
    This test tests the code against the data contained in the American Society
    of Mechanical Engineer's Journal of Fluids Engineering announcement
    "Procedure for Estimation and Reporting of Uncertainty Due to 
    Discretization in CFD Applications." Specifically, this test compares data 
    for the second response quantity, the axial velocity at x/H=8 and y=0.0526.

    References
    ----------
    2008, Procedure for Estimation and Reporting of Uncertainty Due to 
    Discretization in CFD Applications, Journal of Fluids Engineering, 
    130(7): 078001. https://doi.org/10.1115/1.2960953
    """
    N = [18_000, 4_500, 980]
    V = 1 # placeholder volume
    mesh_sizes = utils.mesh_size(V, N, 2)
    phi1 = [10.7880, 10.7250, 10.6050]
    model = dis.Classic(mesh_sizes, phi1)
    assert model.refinement_ratios == approx([2.0, 2.143], abs=0.0005)
    assert model.order["System Response Quantity"] == approx(0.75, abs=0.005)
    assert model.f_est["System Response Quantity"] == approx(10.8801,
                                                             abs=0.00005)
    assert model.uncertainty("System Response Quantity", 0, normalize=True) == approx(0.011, abs=0.0005)


def test_first_and_second_order_literature(roy_2003):
    roy_data = roy_2003[0]
    roy_exact = roy_2003[1]
    model = dis.CustomDiscretizationError(roy_data, model=dis.FirstAndSecondOrder)

    assert model.f_est.values == approx(roy_exact, rel=0.05)
