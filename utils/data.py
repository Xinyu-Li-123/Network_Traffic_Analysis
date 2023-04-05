import pandas as pd 
import json 
import os
from tqdm import tqdm
from scapy.all import IP, ICMP, TCP, UDP
from scapy.all import wrpcap
from scipy.stats import rankdata
import ipaddress
import socket
import struct

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
    if dataset == "caida" and filename == "raw.csv":
        df["time"] = (df["time"] - df["time"].min())
    else:
        df["time"] = (df["time"] - df["time"].min()) / 1e6
    df = df.sort_values(by="time")

    print(f"Number of packets: {len(df)}")
    print(f"Trace duration: {df['time'].max() - df['time'].min()} seconds")

    if verbose:
        print(df)
    
    return df

def IP_str2int(IP_str):
    return int(ipaddress.ip_address(IP_str))

def csv2pcap(input, output):
    """
    Convert a csv file to a pcap file

    :param input: Pandas dataframe of the csv file
    :param output: Path to the output pcap file
    """

    df = input.sort_values(["time"])

    packets = []

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        time = float(row["time"] / 10**6)
        if isinstance(row["srcip"], str):
            srcip = IP_str2int(row["srcip"])
            dstip = IP_str2int(row["dstip"])
            src = socket.inet_ntoa(struct.pack('!L', srcip))
            dst = socket.inet_ntoa(struct.pack('!L', dstip))
        else:
            src = socket.inet_ntoa(struct.pack('!L', row["srcip"]))
            dst = socket.inet_ntoa(struct.pack('!L', row["dstip"]))

        srcport = row["srcport"]
        dstport = row["dstport"]
        proto = row["proto"]
        pkt_len = int(row["pkt_len"])

        try:
            proto = int(proto)
        except BaseException:
            if proto == "TCP":
                proto = 6
            elif proto == "UDP":
                proto = 17
            elif proto == "ICMP":
                proto = 1
            else:
                proto = 0

        ip = IP(src=src, dst=dst, len=pkt_len, proto=proto)
        if proto == 1:
            p = ip / ICMP()
        elif proto == 6:
            tcp = TCP(sport=srcport, dport=dstport)
            p = ip / tcp
        elif proto == 17:
            udp = UDP(sport=srcport, dport=dstport)
            p = ip / udp
        else:
            p = ip

        p.time = time
        p.len = pkt_len
        p.wirelen = pkt_len + 4

        packets.append(p)

    wrpcap(output, packets)
