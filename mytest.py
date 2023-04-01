import numpy as np
import pandas as pd 
import utils.data as data
import utils.stats as stats 
import matplotlib.pyplot as plt

def tmp(dataset, filename):
    df = data.load_data(dataset, filename, verbose=False)
    print(df)

    # compute the packet counts
    timestamps, pkt_counts = stats.byte_count(
        df, time_unit_exp=-2, all_unit=True, verbose=True)

    print(timestamps[:10])
    print(pkt_counts[:10])


    return 

tmp("ugr16", "raw.csv")