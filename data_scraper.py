import os
import json
import numpy as np
import pickle

def scrape_motion_data(directory, filename):
    motion_file = open(directory + filename)
    data = json.load(motion_file)["udpPacket"]
    motion_data = data["mCarMotionData"]
    saved_motion_data = np.zeros((len(motion_data), 6))
    for i in range(len(motion_data)):
        saved_motion_data[i] = [motion_data[i]["mWorldPositionX"], motion_data[i]["mWorldPositionY"],
                                motion_data[i]["mWorldPositionZ"], motion_data[i]["mWorldVelocityX"],
                                motion_data[i]["mWorldVelocityY"], motion_data[i]["mWorldVelocityZ"]] # forward vector, right direction vector (normalize)
    timestamp = data["mHeader"]["mSessionTime"]
    return timestamp, saved_motion_data


def scrape_telemetry_data(directory, filename):
    telemetry_file = open(directory + filename)
    data = json.load(telemetry_file)["udpPacket"]
    telemetry_data = data["mCarTelemetryData"]
    saved_telemetry_data = np.zeros((len(telemetry_data), 3))
    for i in range(len(telemetry_data)):
        saved_telemetry_data[i] = [telemetry_data[i]["mThrottle"], telemetry_data[i]["mSteer"], telemetry_data[i]["mBrake"]]
    timestamp = data["mHeader"]["mSessionTime"]
    return timestamp, saved_telemetry_data

def merge_udp_data(motion_data, telemetry_data):
    udp_data = {}
    mismatch_count = 0
    for timestamp in motion_data.keys():
        try:
            udp_data[timestamp] = np.concatenate((motion_data[timestamp], telemetry_data[timestamp]), axis=1)
        except KeyError:
            mismatch_count += 1
    print("There were a total of", mismatch_count, "file(s) that did not have completed data.")
    return udp_data
         

def scrape_udp_data(motion_directory, telemetry_directory):
    full_motion_data = {}
    full_telemetry_data = {}
    for filename in os.listdir(motion_directory):
        timestamp, motion_data = scrape_motion_data(motion_directory, filename)
        full_motion_data[timestamp] = motion_data
    for filename in os.listdir(telemetry_directory):
        timestamp, telemetry_data = scrape_telemetry_data(telemetry_directory, filename)
        full_telemetry_data[timestamp] = telemetry_data
    udp_data = merge_udp_data(full_motion_data, full_telemetry_data)
    file = open("udp_data.pkl", "wb")
    pickle.dump(udp_data, file)
    file.close()
    
def retrieve_data(timestamp):
    file = open("udp_data.pkl", "rb")
    udp_data = pickle.load(file)
    print(type(udp_data))
    file.close()
    return udp_data[timestamp]


# scrape_udp_data(r"australia_run1/udp_data/motion_packets/", r"australia_run1/udp_data/car_telemetry_packets/")
print(retrieve_data(4.59999609))
