import numpy as np
import pandas as pd
from hurst import compute_Hc
import scipy.signal


def pkt_count(df, time_unit_exp, verbose=True):
    """
    Convert a network trace into a time series of packet counts.
    The time series is aggregated by time unit, which is 10^time_unit_exp seconds.
    For example, if time_unit_exp = -3, then the time series measure the number of packets per 1e-3 seconds.

    :param df: Pandas dataframe of the network traffic
    :param time_unit_exp: Exponent of the time unit, int.
    :param verbose: Print the number of bars in the time series, bool.
    :returns:
      - unique: Time stamps of the time series, numpy array of float.
      - pkt_counts: Packet counts of the time series, numpy array of float.
    """
    T = np.array(df["time"]) * (10**7)   # use int with modulo, avoid floating point modulo
    T = T.astype(int)
    time_unit = 10**(7 + time_unit_exp)  # [1e6, 1e5, 1e4, 1e3] (/1e6 => [1e-1, 1e-2, 1e-3, 1e-4])
    TS = T - T%time_unit  # TS is T aggregated by time unit
    unique, pkt_counts = np.unique(TS, return_counts=True)

    if verbose:
      print("Time unit {:.1e} has {} bars".format(
        time_unit/1e7, len(unique)))

    return unique/1e7, pkt_counts.astype(float)


def byte_count(df, time_unit_exp, verbose=True):
    """
    Similar to pkt_count(), but compute the number of bytes instead of the number of packets.
    """
    T = np.array(df["time"]) * (10**7)   # use int with modulo, avoid floating point modulo
    T = T.astype(int)
    time_unit = 10**(7 + time_unit_exp)  # [1e6, 1e5, 1e4, 1e3] (/1e6 => [1e-1, 1e-2, 1e-3, 1e-4])
    TS = T - T%time_unit  # TS is T aggregated by time unit
    unique, pkt_counts = np.unique(TS, return_counts=True)

    byte_counts = np.zeros_like(pkt_counts)
    if "pkt_len" in df.columns:
      pkt_lens = df["pkt_len"].to_numpy()
    else:
      pkt_lens = df["pkt"].to_numpy()
    pkt_start = 0

    # for loop to compute byte_counts
    for i, pkt_count in enumerate(pkt_counts):
      byte_counts[i] = np.sum(pkt_lens[pkt_start:pkt_start+pkt_count])
      pkt_start += pkt_count 

    if verbose:
      print("Time unit {:.1e} has {} bars".format(
        time_unit/1e7, len(unique)))

    return unique/1e7, byte_counts


def autocorr(data):
    """
    Compute the autocorrelation function of a time series.

    :param data: Time series, numpy array.
    :returns: Autocorrelation function, numpy array.
    """
    # compute the autocorrelation function using numpy.correlate()
    autocorr = scipy.signal.correlate(data, data, mode='full')

    # extract the positive lags of the autocorrelation function
    autocorr = autocorr[len(autocorr)//2:]

    # normalize the autocorrelation function
    autocorr = autocorr / autocorr[0]

    # ignore lag=0
    autocorr[0] = autocorr[1]

    # print the autocorrelation function array
    return autocorr

#TODO: How to normalize to [-1, 1]
def cross_corr(data1, data2):
    """
    Compute the cross-correlation function of two time series.

    :param data1: Time series 1, numpy array.
    :param data2: Time series 2, numpy array.
    :returns: Correlation function, numpy array.
    """
    # compute the correlation function using numpy.correlate()
    ccorr = scipy.signal.correlate(data1, data2, mode='full')

    # extract the positive lags of the correlation function

    # normalize the correlation function
    ccorr = ccorr / np.max(np.abs(ccorr))

    # print the correlation function array
    return ccorr

# TODO: how to caluculate hurst exponent?
def hurst(data):
    """
    Compute the Hurst exponent of a time series.

    :param data: Time series, numpy array.
    :returns: Hurst exponent, float.
    """
    # normalize the data
    data = (data - data.mean()) / data.std()

    # compute the Hurst exponent using the hurst package
    h, _, _ = compute_Hc(data, kind='change', simplified=True)


    # lags = range(2,100)

    # variancetau = []; tau = []

    # for lag in lags: 

    #     #  Write the different lags into a vector to compute a set of tau or lags
    #     tau.append(lag)

    #     # Compute the log returns on all days, then compute the variance on the difference in log returns
    #     # call this pp or the price difference
    #     pp = np.subtract(data[lag:], data[:-lag])
    #     variancetau.append(np.var(pp))

    # # we now have a set of tau or lags and a corresponding set of variances.
    # #print tau
    # #print variancetau

    # # plot the log of those variance against the log of tau and get the slope
    # m = np.polyfit(np.log10(tau),np.log10(variancetau),1)

    # h = m[0] / 2

    # print the Hurst exponent
    return h