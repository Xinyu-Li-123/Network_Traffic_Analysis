import utils.data as data
import utils.stats as stats
import numpy as np
import matplotlib.pyplot as plt

# Load caida data
df = data.load_data("caida", "raw.csv", False)
total_duration = df["time"].max() - df["time"].min()

df = df[df["time"] < total_duration / 20]
total_duration = total_duration / 20

print("Truncated data to 1/20 of original size, total duration: {}".format(total_duration))

# prepare for pca input
dfg = df.groupby(stats.five_tuple)
gks = dfg.groups.keys()

_, _, U, Sigma, VT = stats.flow_pca(dfg, gks, total_duration, -5)
print(U.shape)
