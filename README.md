# Perception Coding Challenge: Ego-Trajectory & BEV Mapping

## Overview
This project estimates the ego-vehicle's trajectory and generates a Bird's Eye View (BEV) map from a sequence of RGB images and depth maps. The solution is divided into two parts:
1.  **Part A**: Ego-trajectory estimation using a static traffic light as a reference.
2.  **Part B**: Enhanced BEV with object tracking (Golf Cart & Barrels).

## Methodology

### Part A: Ego-Trajectory
- **Reference Point**: The traffic light is used as the world origin $(0, 0, Z)$.
- **Approach**: 
    1. Extract the traffic light center $(u, v)$ from provided bounding boxes.
    2. Lookup the 3D position $(X_c, Y_c, Z_c)$ in the camera frame using the `.npz` depth maps.
    3. Compute the ego-vehicle position in the World Frame by inverting the light's position vector, assuming the initial heading aligns with the World X-axis.
- **Assumptions**: 
    - The ground is flat ($Z \approx 0$).
    - The vehicle's orientation remains relatively constant for the short duration (allowing purely translational mapping for the challenge scope).

### Part B: Object Tracking
- **Barrels (Static)**:
    - **Detection**: Color thresholding for **Orange** pixels in RGB ($R>180, G\in[80,180], B<100$).
    - **Mapping**: Valid pixels are projected to 3D using depth maps and transformed to the World Frame.
- **Golf Cart (Dynamic)**:
    - **Detection**: Color thresholding for **White** pixels ($R,G,B > 160$) within the central ROI.
    - **Filtering**: Pixels are filtered by 3D lateral position (within +/- 8m of road center) to reject off-road objects.
    - **Tracking**: The centroid of the valid 3D point cloud is tracked frame-by-frame.

## Coordinate System
- **World Frame**:
    - **Origin**: On the ground, directly below the traffic light.
    - **X-axis**: Aligned with the vehicle's heading at $t=0$.
    - **Z-axis**: Pointing Up.

## Setup & Usage

### 1. Dependencies
Ensure you have Python 3 installed. Install the required libraries:
```bash
pip install numpy matplotlib
```
*Note: `ffmpeg` is optional. If not found, animations will be saved as `.gif` using Pillow.*

### 2. Running Part A
Extracts the ego-trajectory and generates the base visualization.
```bash
python3 analysis_part_a.py
```
**Outputs:**
- `trajectory.png`: Static BEV plot of the ego-vehicle path.
- `trajectory.mp4` / `.gif`: Animation of the vehicle's movement.

### 3. Running Part B (Enhanced)
Runs the enhanced analysis with object detection.
```bash
python3 analysis_part_b.py
```
**Outputs:**
- `trajectory_part_b.png`: Static BEV plot including barrels and the golf cart path.
- `trajectory_part_b.gif`: Animation showing the dynamic scene.
