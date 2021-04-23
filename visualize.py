import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.path as mpltPath
import numpy as np
import pickle
import math
from shapely.geometry import Polygon, Point, LineString
from data_scraper import fetch_data_range, read_scraped_data, fetch_image_from_packet_num
from preprocessing import filter_data_in_view, pointDirectionToPose

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


def visualize_ego_view(new_coordinates, packet_num, image_dir, shape_name, save=False):
    """
    Plots the field of view for the ego vehicle and the positions of cars that are in the field of view
    Positions are in the ego's coordinate system

    Parameters:
    - filename: name of udp file
    - height: height of field of view triangle
    - base: base of field of view triangle
    - timestamp: the timestamp to visualize
    """
    img = fetch_image_from_packet_num(packet_num, image_dir)
    new_coordinates = new_coordinates[packet_num]
    if shape_name == 'triangle':
        base = 90
        height = 30
        p1 = (0,0)
        p2 = (-base/2, height)
        p3 = (base/2, height)
        view_shape = Polygon([p1, p2, p3])
    elif shape_name == 'trapezoid':
        base = 80
        height = 50
        p1 = (-2, 0)
        p2 = (2, 0)
        p3 = (base/2, height)
        p4 = (-base/2, height)
        view_shape = Polygon([p1, p2, p3, p4])
    elif shape_name == 'cone':
        base = 65
        height = 40
        p1 = (-2, 0)
        p2 = (2, 0)
        p3 = (base/2, height)
        p4 = (-base/2, height)
        centerx, centery = 0, height
        radius = base/2
        start_angle, end_angle = 5, 175
        theta = np.radians(np.linspace(start_angle, end_angle, 1000))
        points = [p1, p2, p3]
        for i in range(len(theta)):
            points.append((centerx + radius * np.cos(theta[i]), centery + radius * np.sin(theta[i]) * 0.2))
        points.append(p4)
        view_shape = Polygon(points)
    x1,y1 = view_shape.exterior.xy
    plt.figure(figsize=(30,12))
    plt.subplot(121)
    plt.title("Filtered Data")
    for i in range(len(new_coordinates)):
        if not np.all((new_coordinates[i] == 0)):
            if i == 19:
                plt.plot(new_coordinates[i][0], new_coordinates[i][2], 's', label="Ego Vehicle")
            else:
                plt.plot(new_coordinates[i][0], new_coordinates[i][2], 's', label="Vehicle {0}".format(i+1))
    plt.plot(x1,y1)
    plt.legend()
    plt.xlim(60, -60)
    plt.subplot(122)
    plt.title("In-Game Footage")
    plt.axis("off")
    plt.imshow(img)
    if save:
        plt.savefig("{0}/{1}.png".format(shape_name, packet_num))
    if not save:
        plt.show()