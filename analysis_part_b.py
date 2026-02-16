
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg

def main():
    # Load bbox data (still need light for reference frame)
    csv_file = 'bbox_light.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    frames = []
    light_bboxes = []

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row['frame']))
            light_bboxes.append({
                'x1': int(row['x1']),
                'y1': int(row['y1']),
                'x2': int(row['x2']),
                'y2': int(row['y2'])
            })
    
    # Store data
    num_frames = len(frames)
    print(f"Processing {num_frames} frames for Part B...")
    
    light_in_cam = [] # (x, y, z)
    barrels_in_cam = [] # List of lists of (x,y,z)
    cart_in_cam = [] # (x,y,z) or None
    
    valid_indices = []

    for i in range(num_frames):
        frame_id = frames[i]
        bbox = light_bboxes[i]
        
        # --- 1. Process Traffic Light (Reference) ---
        light_pos = None
        if bbox['x1'] != 0 or bbox['x2'] != 0:
            u = int((bbox['x1'] + bbox['x2']) / 2)
            v = int((bbox['y1'] + bbox['y2']) / 2)
            filename = f"xyz/depth{frame_id:06d}.npz"
            if os.path.exists(filename):
                try:
                    data = np.load(filename)
                    points = data['xyz']
                    H, W, _ = points.shape
                    if v < H and u < W:
                        light_pos = points[v, u][:3]
                        if np.isnan(light_pos[0]) or np.isinf(light_pos[0]) or light_pos[0] == 0:
                            light_pos = None
                except:
                    pass
        light_in_cam.append(light_pos)
        
        # --- 2. Process RGB for Objects ---
        rgb_file = f"rgb/left{frame_id:06d}.png"
        depth_file = f"xyz/depth{frame_id:06d}.npz"
        
        current_barrels = []
        current_cart = None
        
        if os.path.exists(rgb_file) and os.path.exists(depth_file):
            try:
                img = mpimg.imread(rgb_file) # Returns float 0-1 or uint8 0-255
                # Ensure 0-255 uint8 for consistent thresholding
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = (img * 255).astype(np.uint8)
                
                # Load depth once
                data = np.load(depth_file)
                xyz_map = data['xyz'] # (H, W, 4)
                
                # --- Barrels (Orange) ---
                # RGB for Orange is roughly R high, G mid, B low. 
                # E.g. (255, 165, 0)
                # Let's try simple threshold: R > 200, G > 100, B < 100?
                # Actually, standard construction barrels are very orange.
                mask_barrel = (img[:,:,0] > 180) & (img[:,:,1] > 80) & (img[:,:,1] < 180) & (img[:,:,2] < 100)
                
                # Find blobs (simple approach: grid search or connected components if we had cv2/scipy)
                # Without cv2/scipy, we can just take random samples or a crude centroid of the mask.
                # Use mean position of all orange pixels? No, there might be multiple barrels.
                # Let's grid the image and look for density.
                # OR, just take the depth of ALL orange pixels and map them to BEV!
                # That's actually a cool "point cloud" approach for the barrels.
                
                # Let's downsample for speed
                step = 10
                mask_indices = np.where(mask_barrel[::step, ::step])
                # mask_indices is (rows, cols) -> (v, u)
                 
                if len(mask_indices[0]) > 0:
                     # Get corresponding depth indices
                     vs = mask_indices[0] * step
                     us = mask_indices[1] * step
                     
                     # Vectorized lookup?
                     # valid check
                     H, W = xyz_map.shape[:2]
                     valid_mask = (vs < H) & (us < W)
                     vs = vs[valid_mask]
                     us = us[valid_mask]
                     
                     if len(vs) > 0:
                         points = xyz_map[vs, us, :3]
                         # Filter invalid
                         valid_p = ~np.isnan(points[:,0]) & (points[:,0] != 0)
                         current_barrels = points[valid_p]

                # --- Golf Cart (White/Beige) ---
                # It is likely in the center or ahead.
                # White: R > 200, G > 200, B > 200.
                # Limit search to center region to avoid sky/road lines?
                # Sky is blue-ish. Road is grey.
                # Cart is bright white.
                H, W, _ = img.shape
                roi_u_min, roi_u_max = int(W*0.3), int(W*0.7)
                roi_v_min, roi_v_max = int(H*0.4), int(H*0.8) # Ignore sky and immediate bottom
                
                roi = img[roi_v_min:roi_v_max, roi_u_min:roi_u_max]
                mask_cart = (roi[:,:,0] > 220) & (roi[:,:,1] > 220) & (roi[:,:,2] > 220)
                
                # Find centroid of white blob
                ys, xs = np.where(mask_cart)
                if len(xs) > 50: # Threshold size
                    c_x = int(np.mean(xs)) + roi_u_min
                    c_y = int(np.mean(ys)) + roi_v_min
                    
                    # Lookup depth
                    if c_y < H and c_x < W:
                         pt = xyz_map[c_y, c_x, :3]
                         if not np.isnan(pt[0]) and pt[0] != 0:
                             current_cart = pt

            except Exception as e:
                # print(e)
                pass

        barrels_in_cam.append(current_barrels) # Array of points or empty list
        cart_in_cam.append(current_cart)

    # --- Interpolate Light Position (Same as Part A) ---
    light_in_cam_arr = np.array([p if p is not None else [np.nan, np.nan, np.nan] for p in light_in_cam])
    
    # Simple linear interpolation for light
    def interpolate_nans(vals):
        nans = np.isnan(vals)
        if np.all(nans): return vals
        x = lambda z: z.nonzero()[0]
        vals[nans] = np.interp(x(nans), x(~nans), vals[~nans])
        return vals

    x_interp = interpolate_nans(light_in_cam_arr[:, 0].copy())
    y_interp = interpolate_nans(light_in_cam_arr[:, 1].copy())
    z_interp = interpolate_nans(light_in_cam_arr[:, 2].copy())
    
    processed_light = np.stack([x_interp, y_interp, z_interp], axis=1)
    
    if np.all(np.isnan(processed_light)):
        print("No valid light data.")
        return

    # Calculate Rotation to World Frame
    p0 = processed_light[0]
    theta_0 = np.arctan2(p0[1], p0[0])
    c, s = np.cos(-theta_0), np.sin(-theta_0)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) # Rotation from Camera-Aligned to World-Aligned
    
    # Ego Position in World
    # P_ego_w = - (R @ P_light_c)
    # (Assuming P_light_w is origin)
    
    ego_w = []
    for p in processed_light:
        if np.isnan(p).any():
            ego_w.append([np.nan, np.nan, np.nan])
        else:
            ego_w.append(-(R @ p))
    ego_w = np.array(ego_w)
    
    car_x = ego_w[:, 0]
    car_y = ego_w[:, 1]
    
    # --- Transform Objects to World Frame ---
    # P_obj_w = P_ego_w + (R @ P_obj_c)
    # because P_obj_c is relative to ego.
    
    # Process Barrels
    barrels_world_x = []
    barrels_world_y = [] # For all barrels in all frames (visualize accumulation?)
    # Or just visualize current frame's barrels in animation?
    
    # For animation, we need a list of arrays per frame
    barrels_per_frame = []
    
    for i in range(num_frames):
        pts = barrels_in_cam[i]
        if len(pts) > 0:
            # Transform
            # pts is (N, 3)
            # Rotate
            rotated_pts = pts @ R.T # (N,3)
            
            # Add ego pos
            ego_pos = ego_w[i]
            if not np.isnan(ego_pos).any():
                 world_pts = rotated_pts + ego_pos
                 barrels_per_frame.append(world_pts)
                 # Add to accumulator for static plot
                 barrels_world_x.extend(world_pts[:, 0])
                 barrels_world_y.extend(world_pts[:, 1])
            else:
                 barrels_per_frame.append(np.empty((0,3)))
        else:
            barrels_per_frame.append(np.empty((0,3)))

    # Process Cart
    cart_world_x = []
    cart_world_y = []
    
    for i in range(num_frames):
        pt = cart_in_cam[i]
        if pt is not None:
             rotated_pt = R @ pt
             ego_pos = ego_w[i]
             if not np.isnan(ego_pos).any():
                 world_pt = rotated_pt + ego_pos
                 cart_world_x.append(world_pt[0])
                 cart_world_y.append(world_pt[1])
             else:
                 cart_world_x.append(np.nan)
                 cart_world_y.append(np.nan)
        else:
             cart_world_x.append(np.nan)
             cart_world_y.append(np.nan)
             
    # --- Visualization ---

    print("Generating Part B visualization...")
    
    # 1. Static Plot (Accumulated)
    plt.figure(figsize=(12, 12))
    plt.plot(car_x, car_y, label='Ego Trajectory', color='blue')
    plt.scatter([0], [0], color='red', marker='*', s=300, label='Traffic Light (Origin)')
    
    # Scatter all barrels
    # Downsample barrels for plot clarity
    if len(barrels_world_x) > 0:
         plt.scatter(barrels_world_x[::10], barrels_world_y[::10], color='orange', s=5, alpha=0.5, label='Barrels (Static)')
         
    # Cart trajectory
    plt.plot(cart_world_x, cart_world_y, color='green', linewidth=2, label='Golf Cart Path')
    
    plt.xlabel('World X (m)')
    plt.ylabel('World Y (m)')
    plt.title('Part B: Enhanced BEV (Ego + Cart + Barrels)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('trajectory_part_b.png')
    print("Saved trajectory_part_b.png")
    
    # 2. Animation
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Elements
        ego_line, = ax.plot([], [], 'b-', lw=2, label='Ego')
        ego_pt, = ax.plot([], [], 'bo', markersize=8)
        
        cart_line, = ax.plot([], [], 'g-', lw=2, label='Cart')
        cart_pt, = ax.plot([], [], 'go', markersize=8)
        
        barrels_scatter = ax.scatter([], [], c='orange', s=10, alpha=0.6, label='Barrels')
        
        light_marker, = ax.plot([0], [0], 'r*', markersize=15, label='TL')
        
        # Determine bounds
        all_x = list(car_x) + [x for x in cart_world_x if not np.isnan(x)]
        all_y = list(car_y) + [y for y in cart_world_y if not np.isnan(y)]
        
        ax.set_xlim(np.nanmin(all_x) - 10, np.nanmax(all_x) + 10)
        ax.set_ylim(np.nanmin(all_y) - 10, np.nanmax(all_y) + 10)
        ax.set_xlabel('World X (m)')
        ax.set_ylabel('World Y (m)')
        ax.set_title('Part B Animation')
        ax.grid(True)
        # ax.axis('equal') # sometimes messes up limits in animation
        ax.legend()
        
        def init():
            ego_line.set_data([], [])
            ego_pt.set_data([], [])
            cart_line.set_data([], [])
            cart_pt.set_data([], [])
            barrels_scatter.set_offsets(np.empty((0, 2)))
            return ego_line, ego_pt, cart_line, cart_pt, barrels_scatter
        
        def update(frame):
            # Ego
            ego_line.set_data(car_x[:frame+1], car_y[:frame+1])
            ego_pt.set_data([car_x[frame]], [car_y[frame]])
            
            # Cart
            # Handle NaNs in cart path for plotting? Matplotlib handles NaNs by breaking line usually.
            cart_line.set_data(cart_world_x[:frame+1], cart_world_y[:frame+1])
            if not np.isnan(cart_world_x[frame]):
                 cart_pt.set_data([cart_world_x[frame]], [cart_world_y[frame]])
            else:
                 cart_pt.set_data([], [])
                 
            # Barrels (Current Frame observation)
            # To show static map, we could accumulate.
            # But "BEV Scence" usually implies what's visible or the map built so far.
            # Let's show "Map so far" for barrels? Or just current? 
            # Current is less cluttered. Accumulated builds a map. 
            # Let's try Accumulated for valid points.
            
            # Actually, simply showing current detections is safer ensuring they are from valid depths.
            current_b_vals = barrels_per_frame[frame]
            if len(current_b_vals) > 0:
                 barrels_scatter.set_offsets(current_b_vals[:, :2])
            else:
                 barrels_scatter.set_offsets(np.empty((0, 2)))
                 
            return ego_line, ego_pt, cart_line, cart_pt, barrels_scatter
            
        ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=50)
        
        ani.save('trajectory_part_b.gif', writer='pillow', fps=20)
        print("Saved trajectory_part_b.gif")
        
    except Exception as e:
        print(f"Error saving animation: {e}")

if __name__ == "__main__":
    main()
