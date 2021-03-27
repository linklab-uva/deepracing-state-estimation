import matplotlib.pyplot as plt
import numpy as np
import pickle
from data_scraper import fetch_data_range, read_scraped_data

def plot_position_data(filename):
    """
    Plots the position data of all cars (zpos vs. xpos)

    Parameters:
    - filename: filename of udp pickle file
    """
    udp_data = read_data(filename)
    x_position = np.asarray(list(udp_data.values()))[:, :, 0]
    z_position = np.asarray(list(udp_data.values()))[:, :, 2]
    plt.plot(x_position, z_position, 'r.')
    plt.xlabel("X-Position")
    plt.ylabel("Z-Position")

def plot_steering_velocity(filename):
    """
    Plots the velocity and steering angle over time of the ego vehicle
    Only plots first 50 seconds of the session time

    Parameters:
    - filename: filename of udp pickle file
    """
    udp_data = fetch_data_range(0, 50, filename)
    time = np.asarray(list(udp_data.keys()))
    steering = np.zeros(time.shape)
    x_velocity = np.zeros(time.shape)
    y_velocity = np.zeros(time.shape)
    z_velocity = np.zeros(time.shape)
    for i in range(len(time)):
        steering[i] = udp_data[time[i]][0,13]
        x_velocity[i] = udp_data[time[i]][0,3]
        y_velocity[i] = udp_data[time[i]][0,4]
        z_velocity[i] = udp_data[time[i]][0,5]
    velocity = np.sqrt(np.square(x_velocity) + np.square(y_velocity) + np.square(z_velocity))
    fig,ax = plt.subplots()
    ax.plot(time, steering, "r.")
    ax.set_xlabel("Session Time", fontsize=14)
    ax.set_ylabel("Steering Angle", color="red", fontsize=14)
    ax2=ax.twinx()
    ax2.plot(time, velocity, "b.")
    ax2.set_ylabel("Velocity",color="blue",fontsize=14)
    plt.show()

def visualize_waypoint_predictions(labels, predictions):
    """
    Plots the waypoint predictions made by the state estimator with the ground truth labels

    Parameters:
    - labels: array containing ground-truth waypoints
    - predictions: array containing predicted waypoints
    """
    labels_x = np.asarray([])
    labels_z = np.asarray([])
    preds_x = np.asarray([])
    preds_z = np.asarray([])
    for i in range(len(labels) // 3):
        labels_x = np.append(labels_x, labels[3*i])
        labels_z= np.append(labels_z, labels[3*i + 2])
        preds_x = np.append(preds_x, predictions[3*i])
        preds_z = np.append(preds_z, predictions[3*i+2])
    plt.plot(labels_x, labels_z, '.r', label="Ground Truth")
    plt.plot(preds_x, preds_z, '.b', label="Predictions")
    plt.xlabel("X-Position")
    plt.ylabel("Z-Position")
    plt.title("Model Predictions vs. Labels")
    plt.legend()
    plt.show()
