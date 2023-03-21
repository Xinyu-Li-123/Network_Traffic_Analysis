import utils.stats as stats
import utils.data as data
import numpy as np
import matplotlib.pyplot as plt

dataset = "ugr16"
filename = "raw.csv"

# Load the data
df = data.load_data(dataset, filename, verbose=False)

# Compute the packet counts and its hurst exponent
timestamps, packet_counts = stats.pkt_count(df, -0)

# hurst = stats.hurst(packet_counts)

# ts = np.random.normal(0, 1, 10000)
ts = packet_counts
print(ts[:10])

acf = stats.autocorr(ts)
print(f"Autocorrelation function: {acf[:20]}")
print(f"Autocorrelation function: {acf[-20:]}")


plt.plot(acf)
plt.show()

hurst = stats.hurst(ts)
print(f"Hurst exponent: {hurst}")
