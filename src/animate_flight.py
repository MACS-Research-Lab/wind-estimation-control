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

def plot_frame(curr_data, prev_positions, x_range, y_range, z_range):

    prev_positions = np.array(prev_positions)
    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points
    ax.scatter(curr_data['X'], curr_data['Y'], curr_data['Z'])
    ax.scatter(prev_positions[:,0], prev_positions[:,1], prev_positions[:,2], s=1, c=prev_positions[:,3], cmap='RdYlGn', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot')
    
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_zlim(z_range[0], z_range[1])

    # Show plot
    plt.savefig('./figures/test.png')
    # plt.close()
    
    return fig
    
    
def animate_frames(frames):

    # Define the update function for animation
    def update(frame):
        plt.close()
        return frame

    # Create the animation
    ani = FuncAnimation(plt.figure(), update, frames=frames, interval=100)

    # Save the animation as a video file
    ani.save('./figures/animation.mp4', writer='ffmpeg')



if __name__ == "__main__":
    args = parse_command_line()
    
    data = pd.read_csv(args.data_location)
    safe_color = 0
    unsafe_color = 1
    
    offset = 10
    x_range = (min(data['X'])-offset, max(data['X'])+offset)
    y_range = (min(data['Y'])-offset, max(data['Y'])+offset)
    z_range = (min(data['Z'])-offset, max(data['Z'])+offset)
    
    previous_pos = [[data.iloc[0]['X'], data.iloc[0]['Y'], data.iloc[0]['Z'], safe_color]]
    frames = []
    for i, row in tqdm(data.iloc[::50].iterrows(), total=data.iloc[::50].shape[0]):
        frame = plot_frame(row, prev_positions=previous_pos, x_range=x_range, y_range=y_range, z_range=z_range)
        noise = np.random.normal(2, 1)
        was_safe = np.abs(row['TTE']) > 2
        curr_color = safe_color if was_safe else unsafe_color
        previous_pos.append([row['X'], row['Y'], row['Z'], curr_color])
        frames.append(frame)
        
        
    # animate_frames(frames)
    fig, ax = plt.subplots()

    # Function to update the animation
    def update(frame):
        fig.clear()
        fig_new = frames[frame]  # Get the figure object for the current frame
        # fig_new.set_size_inches(fig.get_size_inches())  # Match the size of the original figure
        # fig_new.tight_layout()  # Adjust layout if necessary
        return fig_new

    # Create animation
    ani = FuncAnimation(fig, update, frames=len(frames), interval=200)
    
    ani.save('./figures/animation.mp4', writer='ffmpeg')
        
    
    
