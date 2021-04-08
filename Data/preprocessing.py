import numpy as np
from tqdm import tqdm
import pickle
import math
from shapely.geometry import Polygon, Point
from .data_scraper import read_scraped_data, read_processed_data

def transform_data_for_rnn(filename, buffer_size):
    """
    Attempts to read data from pickle file
    If file not found, performs necessary operations to prepare data for a RNN

    Parameters:
    - filename: the name of the raw udp data file
    - buffer_size: the number of timestamps used for context and future prediction

    Returns: Two 2-D numpy arrays - X_data (features) and y_data (labels)
    Saves: If the files were not previously found, pickle files are saved with the names "X_data.pkl" for features and "y_data.pkl" for labels
    """
    try:
        X_data = read_processed_data("X_data_" + str(buffer_size) + ".pkl")
        y_data = read_processed_data("y_data_" + str(buffer_size) + ".pkl")
    except FileNotFoundError:
        print("Could not find processed data files. Creating new files...")
        udp_data = read_scraped_data(filename)
        X_data = np.empty((0, buffer_size * udp_data.shape[2]))  # (number of samples, buffer size * number of features)
        y_data = np.empty((0, buffer_size * 3))  # (number of samples, buffer size * number of labels)
        for i in tqdm(range(buffer_size - 1, udp_data.shape[0] - buffer_size)):  # iterating over data samples (boundaries of iteration set my number of samples available and buffer size)
            for j in range(udp_data.shape[1]):  # iterating over cars in data sample
                X_data = np.append(X_data, [np.ndarray.flatten(udp_data[(i - buffer_size + 1):i+1, j, :])], axis=0)  # transforms 5 previous feature vectors into a single row (shape = buffer size * number of features)
                y_data = np.append(y_data, [np.ndarray.flatten(udp_data[(i + 1):(i + buffer_size + 1), j, 0:3])], axis=0)  # transforms 5 future position vectors into a single row (shape = buffer size * number of labels)
        # X_data = normalize_vectors(X_data)
        file = open("X_data_" + str(buffer_size) + ".pkl", "wb")
        pickle.dump(X_data, file)
        file.close()
        file = open("y_data_" + str(buffer_size) + ".pkl", "wb")
        pickle.dump(y_data, file)
        file.close()
    finally:
        return X_data, y_data

def normalize_vectors(X_data):
    """
    Normalize feature vectors in filename
    
    Parameters:
    - X_data: numpy array containing preprocessed feature data
    
    Returns: Feature data with normalized vectors
    """
    for i in range(X_data.shape[0]):
        for j in range(X_data.shape[1] // 3):  # dividing by three to account for 3 components per vector
            magnitude = np.sqrt(X_data[i,3*j]**2 + X_data[i,3*j+1]**2 + X_data[i,3*j+2]**2)  # calculates magnitude of vector
            if magnitude != 0:
                X_data[i, 3*j:3*j+3] = X_data[i, 3*j:3*j+3] / magnitude
    return X_data

def filter_data_in_view(filename, height, base):
    """
    Filter data to represent vehicles that are in the field of view of the vehicle
    Note: the vehicle ID of the ego was chosen to be vehicle #20
    
    Parameters:
    - filename: name of udp file
    - height: height of the field of view triangle
    - base: base of the field of view triangle

    Data Format:
    - first row at each index is the vectorized data of the ego vehicle
    - the remaining rows either contain the vectorized data of the other vehicles (if they are in the field of view) or are zero padded if not 
    
    Returns: Filtered data in 3-d matrix form
    """
    udp_data = read_scraped_data(filename)
    filtered_data = np.zeros((udp_data.shape[0], udp_data.shape[1], udp_data.shape[2]))
    for i in tqdm(range(udp_data.shape[0])):
        x = udp_data[i][-1][0]
        z = udp_data[i][-1][2]
        theta = math.atan(udp_data[i][-1][5] / udp_data[i][-1][3])
        phi = math.atan(base/(2*height))
        length = height / math.cos(phi)
        p1 = (x, z)
        p2 = (length*math.cos(theta - phi) + x, length*math.sin(theta - phi) + z)
        p3 = (length*math.cos(theta + phi) + x, length*math.sin(theta + phi) + z)
        view_triangle = Polygon([p1, p2, p3])
        filtered_data[i,0,:] = udp_data[i,-1,:]
        for vehicle_id in range(0, udp_data.shape[1]-1):
            if view_triangle.contains(Point(udp_data[i][vehicle_id][0], udp_data[i][vehicle_id][2])):
                filtered_data[i,vehicle_id+1,:] = udp_data[i,vehicle_id,:]
    return filtered_data