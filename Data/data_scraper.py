import os
import json
import numpy as np
import pickle

def scrape_motion_data(directory, filename):
    """
    Scrapes motion data for all vehicles in a UDP packet
    Used as a helper function for scrape_udp_data (not meant to be called independently)

    Parameters:
    - directory: the directory containing the udp packets
    - filename: the filename of the packet

    Returns: timestamp associated with packet, array of motion data

    Data Format: [x-pos, y-pos, z-pos, x-vel, y-vel, z-vel, x-forwarddir, y-forwarddir, z-forwarddir, x-rightdir, y-rightdir, z-rightdir]
    """
    motion_file = open(directory + filename)
    data = json.load(motion_file)["udpPacket"]
    motion_data = data["mCarMotionData"]
    saved_motion_data = np.zeros((len(motion_data), 12))
    for i in range(len(motion_data)):
        saved_motion_data[i] = [motion_data[i]["mWorldPositionX"], motion_data[i]["mWorldPositionY"], motion_data[i]["mWorldPositionZ"], 
                                motion_data[i]["mWorldVelocityX"], motion_data[i]["mWorldVelocityY"], motion_data[i]["mWorldVelocityZ"],
                                motion_data[i]["mWorldForwardDirX"], motion_data[i]["mWorldForwardDirY"], motion_data[i]["mWorldForwardDirZ"],
                                motion_data[i]["mWorldRightDirX"], motion_data[i]["mWorldRightDirY"], motion_data[i]["mWorldRightDirZ"]]
    timestamp = data["mHeader"]["mSessionTime"]
    return timestamp, saved_motion_data
         
def scrape_udp_data(motion_directory, filename="udp_data.pkl"):
    """
    Collects all udp data within the specified directories

    Parameters:
    - motion_directory: directory containing the motion udp data
    - telemetry_directory: directory containing the telemetry udp data
    - filename: name of pickle file to store scraped data (must be of format *.pkl)
                defaults to "udp_data.pkl"

    Saves: saves dictionary file as a pickle serializable object

    Data Format: [x-pos, y-pos, z-pos, x-vel, y-vel, z-vel, x-forwarddir, y-forwarddir, z-forwarddir, x-rightdir, y-rightdir, z-rightdir]
    """
    try:
        file = open(filename, "rb")
        print("Pickle file already exists")
        file.close()
    except FileNotFoundError:
        udp_data = {}
        if motion_directory[-1] != "/":
            motion_directory += "/"
        for f in os.listdir(motion_directory):
            timestamp, motion_data = scrape_motion_data(motion_directory, f)
            udp_data[timestamp] = motion_data
        file = open(filename, "wb")
        pickle.dump(udp_data, file)
        file.close()
    
def fetch_data(timestamp, filename):
    """
    Opens and reads data from pickle file, returns data at specified timestamp
    If timestamp is not found in data, returns data with closest timestamp to desired time

    Parameters:
    - filename: name of pickle object file
    - timestamp: the desired time at which the data was collected

    Returns: array containing data at specified timestamp range
             if timestamp not in data, closest file to desired to time is returned
    """
    file = open(filename, "rb")
    udp_data = pickle.load(file)
    file.close()
    try:
        return udp_data[timestamp]
    except KeyError:
        data_copy = np.asarray(list(udp_data.keys()))
        idx = (np.abs(data_copy - timestamp)).argmin()
        print("Timestamp not found. Closest value to desired time was used: ", data_copy[idx])
        return udp_data[data_copy[idx]]

def fetch_data_range(start_time, end_time, filename):
    """
    Fetch udp data within specified time range

    Parameters:
    - start_time: start time of desired range
    - end_time: end time of desired range
    - filename: name of udp pickle file

    Returns: dictionary of udp_data with timestamps in desired range
             if there is no data in specified range, and error message is printed
    """
    file = open(filename, "rb")
    udp_data = pickle.load(file)
    file.close()
    times = list(udp_data.keys())
    relevant_times = []
    for i in range(len(times)):
        if times[i] < end_time and times[i] > start_time:
            relevant_times.append(times[i])
    data_copy = {}
    for time in relevant_times:
        data_copy[time] = udp_data[time]
    if data_copy:
        return data_copy
    else:
        print("No timestamps exist within specified range")
        return None

def read_scraped_data(filename):
    """
    Reads udp data stored from scrape_udp_data

    Parameters:
    - filename: name of udp pickle file

    Returns: 3d numpy array of data sorted by timestamp
    """
    file = open(filename, "rb")
    udp_data = pickle.load(file)
    file.close()
    udp_array = np.empty((0, 20, 12)) # 20 cars per packet, 12 variables per car
    for timestamp in sorted(udp_data.keys()):
        udp_array = np.append(udp_array, [udp_data[timestamp]], axis=0)
    return udp_array

def read_processed_data(filename):
    """
    Reads udp data stored from preprocessing

    Parameters:
    - filename: name of udp pickle file

    Returns: 3d numpy array of data formatted for input to a RNN
    """
    file = open(filename, 'rb')
    udp_array = pickle.load(file)
    file.close()
    return udp_array