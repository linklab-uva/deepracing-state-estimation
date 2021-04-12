import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.path as mpltPath
import numpy as np
import pickle
import math
from shapely.geometry import Polygon, Point
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

def visualize_ego_view(filename, height, base, packet_num, image_dir):
    """
    Plots the field of view for the ego vehicle and the positions of cars that are in the field of view
    Note: ego vehicle chosen to be vehicle with ID 20 because it almost always has at least one car in its field of view

    Parameters:
    - filename: name of udp file
    - height: height of field of view triangle
    - base: base of field of view triangle
    - timestamp: the timestamp to visualize
    """
    udp_data = read_scraped_data(filename, packet_num)
    if image_dir[-1] == '/':
        img = mpimg.imread(image_dir + "image_{0}.JPG".format(packet_num))
    else:
        img = mpimg.imread(image_dir + "/image_{0}.JPG".format(packet_num))
    x = udp_data[19][0]
    z = udp_data[19][2]
    theta = math.atan(udp_data[19][8] / udp_data[19][6])
    if udp_data[19][6] < 0:
        theta += math.pi
    phi = math.atan(base/(2*height))
    length = height / math.cos(phi)
    p1 = (x, z)
    p2 = (length*math.cos(theta - phi) + x, length*math.sin(theta - phi) + z)
    p3 = (length*math.cos(theta + phi) + x, length*math.sin(theta + phi) + z)
    view_triangle = Polygon([p1, p2, p3])
    x1,y1 = view_triangle.exterior.xy
    plt.figure(figsize=(30,12))
    plt.subplot(121)
    plt.title("Filtered Data")

    for i in range(udp_data.shape[0]):
        if view_triangle.contains(Point(udp_data[i][0], udp_data[i][2])):
            plt.plot(udp_data[i][0],udp_data[i][2], 's', label="Vehicle {0}".format(i+1))
        elif i == 19:
            plt.plot(udp_data[i][0],udp_data[i][2], 's', label="Ego Vehicle")
        else:
            plt.plot(udp_data[i][0],udp_data[i][2], 's', label="Vehicle {0}".format(i+1))
    plt.plot(x1,y1)
    plt.legend()
    plt.xlim(0, -140)
    plt.subplot(122)
    plt.title("In-Game Footage")
    plt.axis("off")
    plt.imshow(img)

    plt.show()
