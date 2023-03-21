import pandas as pd 
import json 
import os

def load_data(dataset, filename, verbose=True):
    """
    Load csv data of network traffic from the folder data/dataset_name/filename

    :param dataset_name: Name of the dataset
    :param filename: Name of the file
    :return: Pandas dataframe of the network traffic
    """

    # Load the data
    input_folder = os.path.join("data", dataset)
    if not os.path.exists(input_folder):
        raise Exception(f"Input folder does not exist:\n\t{input_folder}")
    
    input_path = os.path.join(input_folder, filename)
    if not os.path.exists(input_path):
        raise Exception(f"Input file does not exist:\n\t{input_path}")

    if not filename.endswith(".csv"):
        raise Exception(f"Input file must be a csv file:\n\t{input_path}")

    print(f"Loading data from:\n\t{input_path}")

    with open(input_path, "r") as f:
        df = pd.read_csv(f)
    
    # TODO: unify column names
    # rename time column 
    if 'time' not in df.columns:
        df = df.rename(columns={'ts': 'time'})
    
    # process time field (divide by 1e6, minus the first time stamp)
    if dataset == "caida":
        df["time"] = (df["time"] - df["time"].min())
    elif dataset == "ugr16":
        df["time"] = (df["time"] - df["time"].min()) / 1e6
    else:
        df["time"] = (df["time"] - df["time"].min()) / 1e6
    df = df.sort_values(by="time")

    print(f"Number of packets: {len(df)}")
    print(f"Trace duration: {df['time'].max() - df['time'].min()} seconds")

    if verbose:
        print(df)
    
    return df


