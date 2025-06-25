import numpy as np

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''
    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''
        def scale_range(input, min_val, max_val):
            input += -(np.min(input))
            input /= np.max(input) / (max_val - min_val)
            input += min_val
            return input

        breakpoints = np.arange(0, buckets + 1) / buckets * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.percentile(expected_array, breakpoints)

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            e_perc = max(e_perc, 0.0001)
            a_perc = max(a_perc, 0.0001)
            return (e_perc - a_perc) * np.log(e_perc / a_perc)

        psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(len(expected_percents))])
        return psi_value

    # Fix initialization
    if len(expected.shape) == 1:
        psi_values = np.array([psi(expected, actual, buckets)])
    else:
        psi_values = np.zeros(expected.shape[1] if axis == 0 else expected.shape[0])

        for i in range(len(psi_values)):
            if axis == 0:
                psi_values[i] = psi(expected[:, i], actual[:, i], buckets)
            elif axis == 1:
                psi_values[i] = psi(expected[i, :], actual[i, :], buckets)

    return psi_values
