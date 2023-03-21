import utils.stats as stats
import utils.data as data
import numpy as np
import matplotlib.pyplot as plt

dataset = "ugr16"
filename = "raw.csv"
# Load the data
df = data.load_data(dataset, filename, verbose=False)

# Compute the packet counts and its hurst exponent
timestamps, timeseries = stats.pkt_count(df, -5)
hurst = stats.hurst(timeseries)
print(f"{filename}: Hurst exponent: {hurst}")

filename = "syn.csv"
# Load the data
df = data.load_data(dataset, filename, verbose=False)
timestamps, timeseries = stats.pkt_count(df, -5)
hurst = stats.hurst(timeseries)
print(f"{filename}: Hurst exponent: {hurst}")
