def my_gof_fun(sim, expected_data=None):
    """
    Define your own goodness of fit function.
    This function takes in a sim, and returns a float (e.g. negative log likelihood) to be maximized.

    sim (starsim.Sim): a simulation object that will have been preconfigured, and run by the Calibration class
    expected_data (Any, but in this example a pandas dataframe): reference data used for comparison with simulation results.
    """
    # Extract and aggregate sim results however we need
    # Extract and aggregate reference data however we need
    # Calculate goodness-of-fit between reference and simulated data
    return np.random.rand(1)  # Just to make the calibration run