import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from matplotlib.animation import FuncAnimation, ArtistAnimation

def parse_command_line():
    parser = argparse.ArgumentParser(description="Process UAV data and create a flight video.")
    parser.add_argument("--data_location", help="Path to the input UAV data")
    parser.add_argument("--video_name", help="Name of the output video file")
    return parser.parse_args()

def plot_uav(ax, X, Y, Z, roll, pitch, yaw):
    roll=pitch=yaw=0
    l = 0.635
    angle = np.pi / 8
    angles = np.arange(0, 2*np.pi, np.pi/4) + np.pi / 2
    motors = l * np.vstack([np.cos(angles), np.sin(angles), np.zeros(8)]).T

    # Define transformation matrices
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Combine rotation matrices
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))

    shift = np.array([X, Y, Z])
    # Transform motor coordinates
    transformed_motors = np.dot(R, motors.T).T + shift


    # Plot motors
    # ax.scatter(transformed_motors[:, 0], transformed_motors[:, 1], transformed_motors[:, 2], color='b')

    # Connect motors to form the UAV body
    for i in range(8):
        ax.plot([X, transformed_motors[(i + 1) % 8, 0]],
                [Y, transformed_motors[(i + 1) % 8, 1]],
                [Z, transformed_motors[(i + 1) % 8, 2]], color='r', linewidth=0.5) 

def plot_frame(fig, curr_data, prev_positions, full_data, x_range, y_range, z_range):
    
    prev_positions = pd.concat([prev_positions, pd.DataFrame({'X': -10000, 'Y': -10000, 'Z': -10000, 'TTE': 0}, index=[0])], ignore_index=True)
    prev_positions = pd.concat([prev_positions, pd.DataFrame({'X': -10000, 'Y': -10000, 'Z': -10000, 'TTE': 100000}, index=[0])], ignore_index=True)
    
    ax = fig.add_subplot(111, projection='3d')
    
    # colors = np.array(np.abs(prev_positions['TTE']) <= 2).astype(int)
    colors = np.clip(np.abs(prev_positions['TTE']), 0, 2)


    # Plot the data points
    # ax.scatter(curr_data['X'], curr_data['Y'], curr_data['Z'], label='UAV')
    scatter_safety = ax.scatter(prev_positions['X'], prev_positions['Y'], prev_positions['Z'], s=1, c=colors, cmap='RdYlGn_r', alpha=0.5)
    ax.scatter(full_data['WPX'], full_data['WPY'], full_data['WPZ'], s=1, c='black', marker='x', label='Waypoints')
    plot_uav(ax, curr_data['X'], curr_data['Y'], curr_data['Z'], curr_data['Roll'], curr_data['Pitch'], curr_data['Yaw'])
    
    cbar = plt.colorbar(scatter_safety)
    cbar.set_label('Deviation (m)')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(z_range[0], z_range[1])
    
    ax.view_init(elev=20)
    # plt.legend()
    
    return fig



if __name__ == "__main__":
    args = parse_command_line()
    
    frameskip = 1
    data = pd.read_csv(args.data_location)
    data = data.iloc[::frameskip].reset_index()
    safe_color = 0
    unsafe_color = 1
    
    offset = 10
    x_range = (min(data['X'])-offset/2, max(data['X'])+offset/2)
    y_range = (min(data['Y'])-offset/2, max(data['Y'])+offset/2)
    z_range = (min(data['Z'])-offset/2, max(data['Z'])+offset/2)
        
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')

    # Function to update the animation
    def update(frame_num):
        fig.clear()
        fig_new = plot_frame(fig, data.iloc[frame_num], data.iloc[:frame_num], data, x_range, y_range, z_range) 
        return fig_new
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=data.shape[0], interval=100)
    
    ani.save('./figures/animation.mp4', writer='ffmpeg')
        
    
    
