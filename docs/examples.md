# Examples

This page provides practical examples of using Space-map for different applications.

## Human Colon Polyp Study

Space-map was used to reconstruct 3D models of human colon tissues using spatial transcriptomics and proteomics data. The dataset included ~2.9M cells from Xenium and ~2.4M cells from CODEX.

```python
import spacemap as sm
import pandas as pd

# Load and process Xenium data
xenium_data = pd.read_csv('xenium_data.csv')
xenium_slices = sm.process_xenium(xenium_data)

# Load and process CODEX data
codex_data = pd.read_csv('codex_data.csv')
codex_slices = sm.process_codex(codex_data)

# Perform registration
mgr = sm.flow.AutoFlowMultiCenter3(xenium_slices + codex_slices)
mgr.alignMethod = "auto"
mgr.affine("DF", show=True)
mgr.ldm_pair(sm.Slice.align1Key, sm.Slice.align2Key, show=True)

# Export results
export = sm.flow.FlowExport(mgr.slices)
export.export_3d_model('colon_polyp_3d_model')
```

## Example 1: Complete Registration Workflow with CSV Data

This example demonstrates a complete workflow for registering multiple tissue sections using cell coordinates from a CSV file.

```python
import spacemap
from spacemap import Slice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load data from CSV file
# The CSV file contains cell coordinates with layer information
df = pd.read_csv("path/to/cells.csv.gz")
groups = df.groupby("layer")

# Organize data by layers
xys = []  # List to store coordinates for each layer
ids = []  # List to store layer IDs

for layer, dff in groups:
    xy = dff[["x", "y"]].values
    ids.append(layer)
    xys.append(xy)

# Step 2: Set up the project
# Define base folder for the project
base = "data/flow"

# Initialize the flow processing system
flow = spacemap.flow.FlowImport(base)

# Initialize with xy coordinates and layer IDs
flow.init_xys(xys, ids)

# Get slice objects for further processing
slices = flow.slices

# Step 3: Perform affine registration for coarse alignment
# Create registration manager
mgr = spacemap.flow.AutoFlowMultiCenter3(slices)

# Set alignment method to automatic
mgr.alignMethod = "auto"

# Perform affine registration
# "DF" specifies the method to use
# show=True displays visualization during the process
mgr.affine("DF", show=True)

# Step 4: Perform LDDMM (Large Deformation Diffeomorphic Metric Mapping)
# This provides high-precision alignment while preserving local structures
mgr.ldm_pair(Slice.align1Key, Slice.align2Key, show=True)

# Step 5: Export the results
export = spacemap.flow.FlowExport(slices)

# Export can be used to save transformed coordinates to CSV
# export.points_csv("aligned_cells.csv")

# Step 6: Visualize the aligned sections
# Create a 3D visualization
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Define colors for different layers
colors = plt.cm.viridis(np.linspace(0, 1, len(slices)))

# Plot each layer
for i, s in enumerate(slices):
    # Get transformed points
    points = s.get_transform_points()
    
    # Add z-coordinate based on layer index
    z_coord = np.ones(len(points)) * i * 10  # 10 units spacing between layers
    points_3d = np.column_stack([points, z_coord])
    
    # Plot points
    ax.scatter(
        points_3d[:, 0], 
        points_3d[:, 1], 
        points_3d[:, 2],
        color=colors[i],
        s=2,
        alpha=0.7,
        label=f"Layer {s.id}"
    )

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Layer)')
ax.legend()
plt.title('3D Reconstruction of Aligned Tissue Sections')
plt.show()

# Optional: Analyze alignment quality
# Calculate average distance between matched points
total_error = 0
total_points = 0

for i in range(len(slices)-1):
    # Get points from adjacent slices
    points1 = slices[i].get_transform_points()
    points2 = slices[i+1].get_transform_points()
    
    # Calculate distances (simplified example)
    # In a real scenario, you would use matched points rather than all points
    min_len = min(len(points1), len(points2))
    points1 = points1[:min_len]
    points2 = points2[:min_len]
    
    # Calculate mean squared error
    mse = np.mean(np.sum((points1 - points2)**2, axis=1))
    total_error += mse * min_len
    total_points += min_len

# Print average alignment error
avg_error = np.sqrt(total_error / total_points) if total_points > 0 else 0
print(f"Average alignment error: {avg_error:.4f} units")
```

## Example 2: Basic Registration of Two Sections

This example demonstrates how to register two adjacent tissue sections containing cell coordinates.

```python
import spacemap as sm
import numpy as np
import matplotlib.pyplot as plt

# 1. Load data
# Simulate two sections with slight displacement and deformation
np.random.seed(42)
n_points = 1000

# Create source points
source_points = np.random.rand(n_points, 2) * 100

# Create target points (slightly shifted and deformed)
angle = np.pi / 15  # Small rotation
scale = 1.1  # Small scaling
translation = np.array([5, 3])  # Translation vector

# Apply transformation to create target points
rotation_matrix = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)]
])
target_points = (scale * (rotation_matrix @ source_points.T).T + 
                 translation + 
                 np.random.normal(0, 2, (n_points, 2)))  # Add noise

# 2. Perform registration
# Step 1: Affine registration
affine_transform = sm.registration.affine_register(
    source_points=source_points,
    target_points=target_points
)

# Apply affine transformation
source_points_affine = sm.affine.transform_points(source_points, affine_transform)

# Step 2: Flow-based fine registration
flow_transform = sm.flow.register(
    source_points=source_points_affine,
    target_points=target_points,
    iterations=100
)

# Apply flow transformation
source_points_final = sm.flow.transform_points(source_points_affine, flow_transform)

# 3. Visualize results
plt.figure(figsize=(15, 5))

# Original points
plt.subplot(1, 3, 1)
plt.scatter(source_points[:, 0], source_points[:, 1], s=5, alpha=0.5, label="Source")
plt.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.5, label="Target")
plt.title("Original Points")
plt.legend()

# After affine registration
plt.subplot(1, 3, 2)
plt.scatter(source_points_affine[:, 0], source_points_affine[:, 1], s=5, alpha=0.5, label="Source (Affine)")
plt.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.5, label="Target")
plt.title("After Affine Registration")
plt.legend()

# After flow-based registration
plt.subplot(1, 3, 3)
plt.scatter(source_points_final[:, 0], source_points_final[:, 1], s=5, alpha=0.5, label="Source (Final)")
plt.scatter(target_points[:, 0], target_points[:, 1], s=5, alpha=0.5, label="Target")
plt.title("After Flow Registration")
plt.legend()

plt.tight_layout()
plt.show()

# 4. Evaluate registration quality
initial_error = np.mean(np.sqrt(np.sum((source_points - target_points)**2, axis=1)))
affine_error = np.mean(np.sqrt(np.sum((source_points_affine - target_points)**2, axis=1)))
final_error = np.mean(np.sqrt(np.sum((source_points_final - target_points)**2, axis=1)))

print(f"Initial average distance error: {initial_error:.2f}")
print(f"After affine registration: {affine_error:.2f}")
print(f"After flow-based registration: {final_error:.2f}")
```

## Example 3: Multi-section Registration with Feature Matching

This example shows how to register multiple sections using both cell coordinates and image features.

```python
import spacemap as sm
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize

# 1. Load data (here we simulate 3 sections)
n_sections = 3
n_points_per_section = 800
section_points = []
section_images = []

np.random.seed(42)

# Simulate section data
for i in range(n_sections):
    # Generate random points with some structure (e.g., cluster pattern)
    centers = np.random.rand(5, 2) * 100
    points = np.vstack([
        np.random.normal(center, scale=10, size=(n_points_per_section // 5, 2))
        for center in centers
    ])
    
    # Add section-specific transformation
    angle = np.random.uniform(-np.pi/10, np.pi/10)
    scale = np.random.uniform(0.9, 1.1)
    translation = np.random.uniform(-10, 10, size=2)
    
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    points = scale * (rotation_matrix @ points.T).T + translation
    section_points.append(points)
    
    # Create a simple image with point density
    img_size = 500
    img = np.zeros((img_size, img_size))
    
    # Scale points to image coordinates
    img_points = np.clip((points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0)) * (img_size-1), 0, img_size-1).astype(int)
    
    # Add points to image with Gaussian blur to simulate staining
    for x, y in img_points:
        img[y, x] = 1
    
    # Add some simulated tissue structures
    for _ in range(20):
        cx, cy = np.random.randint(0, img_size, 2)
        radius = np.random.randint(10, 50)
        for y in range(max(0, cy - radius), min(img_size, cy + radius)):
            for x in range(max(0, cx - radius), min(img_size, cx + radius)):
                if ((x - cx)**2 + (y - cy)**2) < radius**2:
                    img[y, x] = 0.5
    
    # Resize and normalize
    img = resize(img, (200, 200))
    section_images.append(img)

# 2. Register sections
# We'll register each section to the next one, building a 3D stack
transform_db = sm.flow.outputs.TransformDB()

for i in range(n_sections - 1):
    source_points = section_points[i]
    target_points = section_points[i+1]
    source_img = section_images[i]
    target_img = section_images[i+1]
    
    # Find image feature matches
    keypoints_source, keypoints_target, matches = sm.matches.find_matches(
        source_img, target_img, method="sift"
    )
    
    # Affine registration
    affine_transform = sm.registration.affine_register(
        source_points=source_points,
        target_points=target_points
    )
    
    # Apply affine transformation
    source_points_affine = sm.affine.transform_points(source_points, affine_transform)
    
    # Flow-based registration using both points and keypoints
    flow_transform = sm.flow.register(
        source_points=source_points_affine,
        target_points=target_points,
        source_keypoints=keypoints_source,
        target_keypoints=keypoints_target,
        matches=matches,
        iterations=150
    )
    
    # Store transforms
    transform_db.add_transform(
        source_id=i,
        target_id=i+1,
        affine_transform=affine_transform,
        flow_transform=flow_transform
    )
    
    # Transform source points for visualization
    section_points[i] = sm.flow.transform_points(source_points_affine, flow_transform)

# 3. Create 3D model
model_3d = []
for i, points in enumerate(section_points):
    # Add z-coordinate (section index)
    points_3d = np.column_stack([points, np.ones(len(points)) * i * 10])  # 10 units spacing between sections
    model_3d.append(points_3d)

model_3d = np.vstack(model_3d)

# 4. Visualize 3D model
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Color points by section
colors = plt.cm.viridis(np.linspace(0, 1, n_sections))
for i, points in enumerate(section_points):
    points_3d = np.column_stack([points, np.ones(len(points)) * i * 10])
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
               color=colors[i], s=1, alpha=0.7, label=f"Section {i+1}")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Section)')
ax.legend()
plt.title('3D Tissue Model')
plt.show()
```

## Example 3: Working with Real Data

This example demonstrates working with real data from files.

```python
import spacemap as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Assuming you have these data files:
# - section1.csv, section2.csv, etc. with columns 'x', 'y', 'cell_type'
# - section1.tif, section2.tif, etc. as histology images

# 1. Load data
data_dir = "path/to/data"
n_sections = 3  # Number of sections to process
sections_data = []

for i in range(1, n_sections + 1):
    # Load cell coordinates and metadata
    cell_file = os.path.join(data_dir, f"section{i}.csv")
    cells_df = pd.read_csv(cell_file)
    
    # Extract coordinates
    points = cells_df[['x', 'y']].values
    
    # Load histology image if available
    image_file = os.path.join(data_dir, f"section{i}.tif")
    img = None
    if os.path.exists(image_file):
        img = sm.utils.img.load_image(image_file)
    
    sections_data.append({
        "points": points,
        "image": img,
        "metadata": cells_df
    })

# 2. Register adjacent sections
transform_db = sm.flow.outputs.TransformDB()
aligned_sections = [sections_data[0]]  # First section is reference

for i in range(n_sections - 1):
    source = sections_data[i]
    target = sections_data[i+1]
    
    print(f"Registering section {i+1} to {i+2}...")
    
    # Perform affine registration
    affine_transform = sm.registration.affine_register(
        source_points=source["points"],
        target_points=target["points"]
    )
    
    # Apply affine transformation
    source_points_affine = sm.affine.transform_points(source["points"], affine_transform)
    
    # Perform flow-based registration if images are available
    if source["image"] is not None and target["image"] is not None:
        # Find image feature matches
        keypoints_source, keypoints_target, matches = sm.matches.find_matches(
            source["image"], target["image"], method="loftr"
        )
        
        # Perform flow registration using both points and keypoints
        flow_transform = sm.flow.register(
            source_points=source_points_affine,
            target_points=target["points"],
            source_keypoints=keypoints_source,
            target_keypoints=keypoints_target,
            matches=matches,
            iterations=200
        )
    else:
        # Use only points for registration
        flow_transform = sm.flow.register(
            source_points=source_points_affine,
            target_points=target["points"],
            iterations=200
        )
    
    # Store transforms
    transform_db.add_transform(
        source_id=i,
        target_id=i+1,
        affine_transform=affine_transform,
        flow_transform=flow_transform
    )
    
    # Apply transform to the target section
    aligned_section = dict(target)
    aligned_section["points"] = sm.flow.transform_points(
        target["points"], 
        flow_transform
    )
    
    # Store aligned section
    aligned_sections.append(aligned_section)

# 3. Create 3D model with cell type information
model_3d = []
cell_types = []

for i, section in enumerate(aligned_sections):
    # Add z-coordinate (section index)
    points = section["points"]
    points_3d = np.column_stack([points, np.ones(len(points)) * i * 10])
    model_3d.append(points_3d)
    
    # Extract cell types if available
    if "metadata" in section and "cell_type" in section["metadata"].columns:
        cell_types.extend(section["metadata"]["cell_type"].tolist())

model_3d = np.vstack(model_3d)

# 4. Visualize 3D model colored by cell type
if cell_types:
    unique_types = list(set(cell_types))
    type_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
    type_to_color = {t: type_colors[i] for i, t in enumerate(unique_types)}
    
    point_colors = [type_to_color[t] for t in cell_types]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each cell type separately for legend
    for i, cell_type in enumerate(unique_types):
        mask = np.array(cell_types) == cell_type
        if np.any(mask):
            ax.scatter(
                model_3d[mask, 0], 
                model_3d[mask, 1], 
                model_3d[mask, 2],
                color=type_colors[i],
                s=3,
                alpha=0.7,
                label=cell_type
            )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Section)')
    ax.legend()
    plt.title('3D Tissue Model with Cell Types')
    plt.show()

# 5. Save the model
output_dir = os.path.join(data_dir, "results")
os.makedirs(output_dir, exist_ok=True)

# Save transformation database
transform_db.save(os.path.join(output_dir, "transforms.pkl"))

# Save 3D model
np.save(os.path.join(output_dir, "model_3d.npy"), model_3d)

# Save as CSV with metadata
if cell_types:
    model_df = pd.DataFrame({
        'x': model_3d[:, 0],
        'y': model_3d[:, 1],
        'z': model_3d[:, 2],
        'cell_type': cell_types
    })
    model_df.to_csv(os.path.join(output_dir, "model_3d.csv"), index=False)

print(f"Results saved to {output_dir}")
```

These examples should help you get started with Space-map. For more details on specific functions and features, refer to the [API Reference](api.md) and [Detailed Usage Guide](usage.md). 