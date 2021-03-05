import numpy as np
from tqdm import tqdm
import pickle
from data_scraper import read_scraped_data, read_processed_data

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
        X_data = read_processed_data("X_data.pkl")
        y_data = read_processed_data("y_data.pkl")
    except FileNotFoundError:
        print("Could not find processed data files. Creating new files...")
        udp_data = read_scraped_data(filename)
        X_data = np.empty((0, buffer_size * udp_data.shape[2]))  # (number of samples, buffer size * number of features)
        y_data = np.empty((0, buffer_size * 3))  # (number of samples, buffer size * number of labels)
        for i in tqdm(range(buffer_size - 1, udp_data.shape[0] - buffer_size)):  # iterating over data samples (boundaries of iteration set my number of samples available and buffer size)
            for j in range(udp_data.shape[1]):  # iterating over cars in data sample
                X_data = np.append(X_data, [np.ndarray.flatten(udp_data[(i - buffer_size + 1):i+1, j, :])], axis=0)  # transforms 5 previous feature vectors into a single row (shape = buffer size * number of features)
                y_data = np.append(y_data, [np.ndarray.flatten(udp_data[(i + 1):(i + buffer_size + 1), j, 0:3])], axis=0)  # transforms 5 future position vectors into a single row (shape = buffer size * number of labels)
        file = open("X_data.pkl", "wb")
        pickle.dump(X_data, file)
        file.close()
        file = open("y_data.pkl", "wb")
        pickle.dump(y_data, file)
        file.close()
    finally:
        return X_data, y_data