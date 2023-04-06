import numpy as np
import pandas as pd
# from hurst import compute_Hc
import scipy.signal
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.sparsefuncs import mean_variance_axis
from scipy.sparse import lil_matrix, csr_matrix, issparse


five_tuple = ["srcip", "srcport", "dstip", "dstport", "proto"]


def pkt_count(df, time_unit_exp, total_duration, all_unit=False, verbose=True):
    """
    Convert a network trace into a time series of packet counts.
    The time series is aggregated by time unit, which is 10^time_unit_exp seconds.
    For example, if time_unit_exp = -3, then the time series measure the number of packets per 1e-3 seconds.

    :param df: Pandas dataframe of the network traffic
    :param time_unit_exp: Exponent of the time unit, int.
    :param all_unit: 
        if set true, return pkt counts of all time units; 
        otherwise, return pkt counts at least 1 and its correponding timestamps
        e.g.
        df = pd.DataFrame({"time": [0.010, 0.022, 0.035, 0.213]})
        pkt_count(df, -1, all_unit=False)   # [0, 0.2], [3, 1]
        pkt_count(df, -1, all_unit=False)   # [0, 0.1, 0.2], [3, 0, 1]
    :param verbose: Print the number of bars in the time series, bool.
    :returns:
      - unique: Time stamps of the time series, numpy array of float.
      - pkt_counts: Packet counts of the time series, numpy array of float.
    """
    T = np.array(df["time"]) * 1e7   # use int with modulo, avoid floating point modulo
    # T = T.astype(int)
    time_unit = 10**(7 + time_unit_exp)  # [1e6, 1e5, 1e4, 1e3] (/1e6 => [1e-1, 1e-2, 1e-3, 1e-4])
    TS = T - T%time_unit  # TS is T aggregated by time unit
    unique, pkt_counts = np.unique(TS, return_counts=True)

    if verbose:
        print("Time unit {:.1e} has {} bars".format(
            time_unit/1e7, len(unique)))
    
    if not all_unit:
        return unique/1e7, pkt_counts.astype(float)
    else:
        unique = unique/1e7
        ts = np.arange(0, total_duration+10**time_unit_exp, 10**time_unit_exp)
        all_pkt_counts = np.zeros(len(ts))
        print(unique.shape)
        print(total_duration)
        print(ts.shape)
    
        for i, t in enumerate(unique):
            ai = int(t*(10**(-time_unit_exp)))
            all_pkt_counts[ai] = pkt_counts[i]

        return ts, all_pkt_counts


def byte_count(df, time_unit_exp, total_duration, all_unit=False, verbose=True):
    """
    Similar to pkt_count(), but compute the number of bytes instead of the number of packets.
    """
    T = np.array(df["time"]) * (10**7)   # use int with modulo, avoid floating point modulo
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

    if not all_unit:
        return unique/1e7, byte_counts
    else:
        unique = unique/1e7
        ts = np.arange(0, total_duration+10**time_unit_exp, 10**time_unit_exp)
        all_byte_counts = np.zeros(len(ts))
        print(ts.shape) 
        for i, t in enumerate(unique):
            ai = int(t*(10**(-time_unit_exp)))
            all_byte_counts[ai] = byte_counts[i]

        return ts, all_byte_counts


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

def pearson_corr(flows):
    """
    Compute the Pearson correlation coefficient of a set of flows.
    
    :param flows: Flows, list of numpy arrays, shape = (num_t, num_flows)
    :return pcorr: Pearson correlation coefficient, numpy array, shape = (num_flows, num_flows)
            pcorr[i, j] = Pearson correlation coefficient between flow i and flow j
    
    """
    return np.corrcoef(flows, rowvar=False)



# TODO: how to caluculate hurst exponent?
def hurst(data):
    """
    Compute the Hurst exponent of a time series.

    :param data: Time series, numpy array.
    :returns: Hurst exponent, float.
    """
    raise NotImplemented("Hurst exponent is not properly implemented yet.")
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

def flow_pca(dfg, gks, total_duration, time_unit_exp=-6, n_components=9, rate_type="packet", verbose=True):
    """
    Convert network traffic into a od_flow matrix (origin-destination flow) and use truncated SVD to reduce dimension

    We treat all records with the same 5-tuple as a flow.

    We construct a matrix X of shape (T, N) where T is #time intervals and N is #flows.

    X[:, j] is the ith flow (a univariate time series, e.g. packet counts across time of a particular 5-tuple)
    X[i, j] is a measurement in time series (e.g. packet count)

    Performing a SVD on X (treat each column as a vector, compute the top `n_components` eigenvectors).

    Using eigenvectors to represent X can only preserve limited amount of information. The percentage of information
        preserved can be calculated by summing the variance of each eigenvector (each eigenvector accounts for a portion
        of variance).
    
    Parameters
    ----------

    :param dfg: A pandas groupby object where each group is a flow.
    :param gks: The group keys of needed groups (e.g. group keys of flows with size <= 3).
    :param total_duration: The total duration of the traffic in seconds.
    :param time_unit_exp: The exponent of the time unit. For example, if time_unit_exp=-6, then the time unit is 1e-6 seconds.
    :param n_components: The number of components to keep.

    :returns:
        od_flows: The od_flow matrix, shape (T, N)
        trunc_od_flows: The truncated od_flow matrix, shape (T, n_components)
        U: The left singular vectors, shape (T, n_components)
        Sigma: The singular values, shape (n_components,)
        VT: The right singular vectors, shape (n_components, N)

    """
    def pkt_count_flow_pca(df, time_unit_exp, rate_type):
        T = np.array(df["time"]) * (10**7)   # use int with modulo, avoid floating point modulo
        T = T.astype(int)
        time_unit = 10**(7 + time_unit_exp)  # [1e6, 1e5, 1e4, 1e3] (/1e6 => [1e-1, 1e-2, 1e-3, 1e-4])
        TS = T - T%time_unit  # TS is T aggregated by time unit
        unique, pkt_counts = np.unique(TS, return_counts=True)
        if rate_type == "byte":
            byte_counts = np.zeros_like(pkt_counts)
            pkt_lens = df["pkt_len"].to_numpy()
            pkt_start = 0
            # for loop to compute byte_counts
            for i, pkt_count in enumerate(pkt_counts):
                byte_counts[i] = np.sum(pkt_lens[pkt_start:pkt_start+pkt_count])
                pkt_start += pkt_count 

        return TS, unique/1e7, pkt_counts

    num_t = int(total_duration / 10**time_unit_exp)
    num_flow = len(gks)

    print("Time unit={:.2e} with {} time intervals and {} flows".format(
        10**time_unit_exp, 
        num_t, 
        num_flow))

    # od_flows = np.zeros((num_t, num_flow))
    od_flows = lil_matrix((num_t, num_flow))
    print(od_flows.shape)

    # bins = np.arange(0, total_duration, 10**time_unit_exp)  

    # for j, (gk, g) in enumerate(dfg):
    for j, gk in enumerate(gks):
        g = dfg.get_group(gk)
        if j==0 or (j+1)%1000==0 or j+1==num_flow:
            print(f"\r{j+1}/{num_flow}", end="")
        ts, unique, pkts = pkt_count_flow_pca(
            g, time_unit_exp, rate_type)
        i = 0
        prev_t = ts[0]
        for t in ts:
            if prev_t != t:
                i += 1
            od_flows[t//(10**(7+time_unit_exp))-1, j] = pkts[i]
            prev_t = t
    print()

    U, Sigma, VT = randomized_svd(csr_matrix(od_flows), 
                                n_components=n_components,
                                n_iter=5,
                                random_state=None)
    od_flows = od_flows.toarray()
    trunc_od_flows = U @ np.diag(Sigma) @ VT

    # compute explained variance
    explained_variance_ = exp_var = np.var(trunc_od_flows, axis=0)
    if issparse(trunc_od_flows):
        _, full_var = mean_variance_axis(od_flows, axis=0)
        full_var = full_var.sum()
    else:
        full_var = np.var(od_flows, axis=0).sum()
    explained_variance_ratio_ = exp_var / full_var

    return od_flows, trunc_od_flows, U, Sigma, VT, explained_variance_, explained_variance_ratio_


