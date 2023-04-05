import numpy as np
import pandas as pd 
import utils.data as data
import utils.stats as stats 
import matplotlib.pyplot as plt

# Load the data
dataset = "caida"
filename = "raw.csv"
df = data.load_data(dataset, filename, verbose=False)

# find max flow
dfg = df.groupby(stats.five_tuple)
flowsizes = dfg.size()
max_flow = flowsizes.idxmax()
print(f"Max flow: {max_flow}, size: {flowsizes[max_flow]}")
df_max_flow = dfg.get_group(max_flow)
print(df_max_flow)

# save as pcap
output = "data/caida_maxflow/raw.pcap"
data.csv2pcap(df_max_flow, output)