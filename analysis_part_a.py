
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv
from matplotlib.animation import FuncAnimation
from PIL import Image

def main():
    # Load bbox data
    csv_file = 'bbox_light.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    frames = []
    bboxes = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row['frame']))
            bboxes.append({
                'x1': int(row['x1']),
                'y1': int(row['y1']),
                'x2': int(row['x2']),
                'y2': int(row['y2'])
            })
    
    # Store trajectory points
    
    # Check if xyz dir exists
    if not os.path.exists('xyz'):
        print("Error: xyz directory not found.")
        return

    num_frames = len(frames)
    print(f"Processing {num_frames} frames...")
    
    # List to hold raw relative coords (Light in Camera Frame)
    light_in_cam = [] # (x, y, z)
    valid_indices = []

    for i in range(num_frames):
        frame_id = frames[i]
        bbox = bboxes[i]
        
        # Check for invalid bbox (zeros)
        if bbox['x1'] == 0 and bbox['x2'] == 0:
            light_in_cam.append(None)
            continue
            
        # Get center of bbox
        u = int((bbox['x1'] + bbox['x2']) / 2)
        v = int((bbox['y1'] + bbox['y2']) / 2)
        
        # Load depth
        filename = f"xyz/depth{frame_id:06d}.npz"
        
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            light_in_cam.append(None)
            continue
            
        try:
            data = np.load(filename)
            points = data['xyz'] # (H, W, 3)
            
            # Check bounds
            H, W, _ = points.shape
            if v >= H or u >= W:
                print(f"Index out of bounds for frame {frame_id}: {u}, {v}")
                light_in_cam.append(None)
                continue
                
            # Extract XYZ
            x_c, y_c, z_c = points[v, u][:3]
            
            # Filter naive invalid depths
            if np.isnan(x_c) or np.isinf(x_c) or x_c == 0:
                 light_in_cam.append(None)
            else:
                 light_in_cam.append([x_c, y_c, z_c])
                 valid_indices.append(i)
                 
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            light_in_cam.append(None)

    # Interpolate
    # Convert to numpy array with NaNs
    light_in_cam_arr = np.array([p if p is not None else [np.nan, np.nan, np.nan] for p in light_in_cam])
    
    # Simple linear interpolation for each column
    x_vals = light_in_cam_arr[:, 0]
    y_vals = light_in_cam_arr[:, 1]
    z_vals = light_in_cam_arr[:, 2]
    
    def interpolate_nans(vals):
        nans = np.isnan(vals)
        if np.all(nans): return vals # All nan
        
        # Create x array for interpolation
        x = lambda z: z.nonzero()[0]
        vals[nans] = np.interp(x(nans), x(~nans), vals[~nans])
        return vals

    x_interp = interpolate_nans(x_vals.copy())
    y_interp = interpolate_nans(y_vals.copy())
    z_interp = interpolate_nans(z_vals.copy())
    
    processed_xyz = np.stack([x_interp, y_interp, z_interp], axis=1)
    
    if np.all(np.isnan(processed_xyz)):
        print("No valid data extracted.")
        return

    # Initial position check
    p0 = processed_xyz[0]
    print(f"Initial Light Pos in Cam: {p0}")
    
    # Calculate initial angle in Camera X-Y plane
    theta_0 = np.arctan2(p0[1], p0[0])
    print(f"Initial offset angle: {np.degrees(theta_0):.2f} degrees")
    
    # Rotation matrix to rotate vectors in XY plane by -theta_0
    # To align the initial vector with the X axis.
    c, s = np.cos(-theta_0), np.sin(-theta_0)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    aligned_light_pos = []
    for p in processed_xyz:
        aligned_light_pos.append(R @ p)
        
    aligned_light_pos = np.array(aligned_light_pos)
    
    # Car position is negative of Light position (relative to origin under light)
    car_x = -aligned_light_pos[:, 0]
    car_y = -aligned_light_pos[:, 1]
    
    # Filter out any outliers or smooth? 
    print(f"Car X range: {np.min(car_x):.2f} to {np.max(car_x):.2f}")
    print(f"Car Y range: {np.min(car_y):.2f} to {np.max(car_y):.2f}")
    
    # Optional: Moving average
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
        
    # Apply slight smoothing if needed, but let's keep raw first.
    
    # --- Visualization ---
    
    # 1. Static Plot (PNG)
    plt.figure(figsize=(10, 10))
    plt.plot(car_x, car_y, label='Car Trajectory')
    plt.scatter([0], [0], color='red', marker='*', s=200, label='Traffic Light (Origin)')
    
    # Mark start and end
    plt.scatter([car_x[0]], [car_y[0]], color='green', label='Start')
    plt.scatter([car_x[-1]], [car_y[-1]], color='blue', label='End')

    plt.xlabel('World X (m)')
    plt.ylabel('World Y (m)')
    plt.title('Ego-Vehicle BEV Trajectory')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('trajectory.png')
    print("Saved trajectory.png")
    
    # 2. Animation (MP4)
    # Check if we can save mp4
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        line, = ax.plot([], [], 'b-', lw=2)
        point, = ax.plot([], [], 'bo')
        light_marker, = ax.plot([0], [0], 'r*', markersize=15)
        
        ax.set_xlim(np.min(car_x) - 5, np.max(car_x) + 5)
        ax.set_ylim(np.min(car_y) - 5, np.max(car_y) + 5)
        ax.set_xlabel('World X (m)')
        ax.set_ylabel('World Y (m)')
        ax.set_title('Ego-Vehicle BEV Trajectory')
        ax.grid(True)
        ax.axis('equal')
        
        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point,
        
        def update(frame):
            # frame is index
            if frame % 10 == 0:
                print(f"Update frame {frame}")
            line.set_data(car_x[:frame+1], car_y[:frame+1])
            point.set_data([car_x[frame]], [car_y[frame]]) 
            return line, point,
            
        ani = FuncAnimation(fig, update, frames=len(car_x), init_func=init, blit=False, interval=50) # 20fps
        
        ani.save('trajectory.mp4', writer='ffmpeg', fps=20)
        print("Saved trajectory.mp4")
    except Exception as e:
        print(f"Warning: Could not save mp4. {e}")
        try:
            plt.close(fig) # Close old figure
            
            # Use explicit Agg backend to guarantee off-screen rendering
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            fig = Figure(figsize=(10, 10))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            
            print("Starting manual GIF generation with FigureCanvasAgg (Brute Force)...")
            frames_img = []
            
            # Pre-calculate limits
            all_x = np.concatenate([car_x, [0]])
            all_y = np.concatenate([car_y, [0]])
            xlims = (np.min(all_x) - 5, np.max(all_x) + 5)
            ylims = (np.min(all_y) - 5, np.max(all_y) + 5)

            for f in range(len(car_x)):
                if f % 50 == 0:
                    print(f"Rendering frame {f}/{len(car_x)}")
                
                ax.clear()
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                ax.set_xlabel('World X (m)')
                ax.set_ylabel('World Y (m)')
                ax.set_title('Ego-Vehicle BEV Trajectory')
                ax.grid(True)
                ax.set_aspect('equal')
                
                # Re-plot
                ax.plot(car_x[:f+1], car_y[:f+1], 'b-', lw=2)
                ax.plot([car_x[f]], [car_y[f]], 'bo')
                ax.plot([0], [0], 'r*', markersize=15)
                
                # Redraw
                canvas.draw()
                
                # Convert canvas to image
                img = np.asarray(canvas.buffer_rgba()).copy()
                img = Image.fromarray(img)
                
                frames_img.append(img)
            
            if frames_img:
                frames_img[0].save(
                    'trajectory.gif',
                    save_all=True,
                    append_images=frames_img[1:],
                    duration=50, # 50ms per frame = 20fps
                    loop=0
                )
                print(f"Saved trajectory.gif with {len(frames_img)} frames manually.")
            else:
                 print("Error: No frames generated.")

        except Exception as e2:
            print(f"Error saving gif: {e2}")
            import traceback
            traceback.print_exc()
        except Exception as e2:
            print(f"Error saving gif: {e2}")

if __name__ == "__main__":
    main()
