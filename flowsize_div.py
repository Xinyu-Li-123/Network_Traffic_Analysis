import utils.stats as stats
import utils.data as data
import numpy as np
import matplotlib.pyplot as plt

def flowsize_demo(time_unit_exp=1):
    df = data.load_data("caida", "raw.csv", verbose=False)
    # timestamps, timeseries = stats.pkt_count(df, time_unit_exp)
    flowsizes = df.groupby(stats.five_tuple).size()
    
    # plot cdf of flow sizes
    flowsizes = flowsizes.to_numpy()

    a, b = np.histogram(flowsizes, bins=100)
    flowsizes_cdf = np.cumsum(a)
    flowsizes_cdf = flowsizes_cdf / flowsizes_cdf[-1]

    flowsizes_pdf = a / np.sum(a)

    # plot cdf of flow sizes in log scale
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    axes[0].plot(flowsizes_cdf, label="raw")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Flow Size (pkts)")
    axes[0].set_ylabel("CDF")
    axes[0].set_title("Flow Size CDF")

    axes[1].plot(flowsizes_pdf, label="raw")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Flow Size (pkts)")
    axes[1].set_ylabel("PDF")
    axes[1].set_title("Flow Size PDF")

    df = data.load_data("caida", "syn.csv", verbose=False)
    # timestamps, timeseries = stats.pkt_count(df, time_unit_exp)
    flowsizes = df.groupby(stats.five_tuple).size()
    
    # plot cdf of flow sizes
    flowsizes = flowsizes.to_numpy()

    a, b = np.histogram(flowsizes, bins=100)
    flowsizes_cdf = np.cumsum(a)
    flowsizes_cdf = flowsizes_cdf / flowsizes_cdf[-1]

    flowsizes_pdf = a / np.sum(a)

    axes[0].plot(flowsizes_cdf, label="syn")

    axes[1].plot(flowsizes_pdf, label="syn")

    # adjust padding
    fig.tight_layout(pad=1.0)

    plt.show()

def flowsize_div(time_unit_exp=1):
    df = data.load_data("ugr16", "raw.csv", verbose=False)
    total_duration = df["time"].max() - df["time"].min()
    # timestamps, timeseries = stats.pkt_count(df, time_unit_exp)
    dfg = df.groupby(stats.five_tuple)
    flowsizes = dfg.size().reset_index().rename(columns={0: "size"})
    small_flowheader = flowsizes[flowsizes["size"] <= 10][stats.five_tuple]
    big_flowheader = flowsizes[flowsizes["size"] > 10][stats.five_tuple]
    print(small_flowheader.columns)
    _, _, U, _, _ = stats.flow_pca(
        dfg, 
        small_flowheader, 
        total_duration, time_unit_exp, n_components=2, verbose=False)

    print(U.shape)
    


    



# flowsize_demo(-1)
flowsize_div(-0)




    