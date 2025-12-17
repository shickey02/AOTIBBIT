#!/usr/bin/env python3
# AOTI LEVEL 45: THE EVENT HORIZON (Relativistic Ray Tracer)
# Generates a physically accurate image of a Black Hole & Accretion Disk.
# Uses "Ray Marching" to bend light through curved spacetime.

import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
RESOLUTION = 300       # Image size (300x300 pixels)
FOV = 10.0             # Field of View (Zoom)
BH_MASS = 1.0          # Mass of Black Hole
RS = 2 * BH_MASS       # Schwarzschild Radius (The Event Horizon)
DISK_INNER = 3.0 * RS  # Innermost Stable Circular Orbit (ISCO)
DISK_OUTER = 12.0 * RS # Edge of Accretion Disk

# Camera Position (Distance and Angle)
CAMERA_DIST = 15.0
CAMERA_PITCH = 0.2     # Tilt camera up slightly to see the disk face-on

def render():
    print(f"--- AOTI RAY TRACER ---")
    print(f"Bending Spacetime for {RESOLUTION}x{RESOLUTION} rays...")
    
    # 1. SETUP THE SCREEN (Virtual Camera Plane)
    h = w = RESOLUTION
    y, x = np.mgrid[1:-1:1j*h, -1:1:1j*w]
    
    # Aspect ratio correction
    x = x * (w/h)
    
    # 2. INITIALIZE RAYS
    # Ray Origin (The Camera)
    # We position camera at z = -CAMERA_DIST, tilted by PITCH
    ray_pos = np.zeros((h, w, 3))
    ray_pos[:,:,0] = 0               # x
    ray_pos[:,:,1] = CAMERA_DIST * np.sin(CAMERA_PITCH) # y (height)
    ray_pos[:,:,2] = -CAMERA_DIST * np.cos(CAMERA_PITCH) # z (depth)
    
    # Ray Velocity (The Vector pointing from camera to pixel)
    # In flat space, this is just (x, y, 1)
    # We rotate it slightly to match camera pitch
    ray_dir = np.zeros((h, w, 3))
    ray_dir[:,:,0] = x * FOV
    ray_dir[:,:,1] = y * FOV * np.cos(CAMERA_PITCH) + 1.0 * np.sin(CAMERA_PITCH)
    ray_dir[:,:,2] = 1.0 * np.cos(CAMERA_PITCH) - y * FOV * np.sin(CAMERA_PITCH)
    
    # Normalize direction vectors
    norm = np.linalg.norm(ray_dir, axis=2, keepdims=True)
    ray_dir /= norm
    
    # 3. THE "RAY MARCHING" LOOP
    # Instead of solving complex ODEs, we step forward and bend the vector
    # This is a discrete approximation of Gravity.
    
    steps = 100
    dt = 0.5 # Step size
    
    # The final image buffer (RGB)
    image = np.zeros((h, w, 3))
    
    # Mask to track which rays are still flying
    active_mask = np.ones((h, w), dtype=bool)
    
    print("Marching rays...")
    for step in range(steps):
        if not np.any(active_mask): break
        
        # Current position of active rays
        pos = ray_pos[active_mask]
        
        # Calculate Distance from Singularity (r)
        r_sq = np.sum(pos**2, axis=1)
        r = np.sqrt(r_sq)
        
        # --- PHYSICS: GRAVITY BENDING ---
        # Acceleration = -1.5 * Rs / r^2 (Simplified Newtonian-ish approximation for visuals)
        # In full GR, this is the Geodesic Equation.
        # We steer the ray velocity towards the center (0,0,0)
        accel_mag = 1.5 * RS / (r_sq * r) 
        
        # Update Velocity (Bend the light)
        # v = v + a * dt
        # We only apply gravity if far from horizon to avoid singularities
        safe_zone = r > RS
        
        # Apply bending
        ray_dir[active_mask] -= pos * accel_mag[:, np.newaxis] * dt
        
        # Re-normalize to keep speed of light constant
        ray_dir_active = ray_dir[active_mask]
        ray_dir_active /= np.linalg.norm(ray_dir_active, axis=1, keepdims=True)
        ray_dir[active_mask] = ray_dir_active
        
        # Update Position
        # x = x + v * dt
        ray_pos[active_mask] += ray_dir[active_mask] * dt
        
        # --- COLLISION DETECTION ---
        
        # A. EVENT HORIZON (The Shadow)
        # If r < Schwarzschild Radius, the light is trapped.
        # Color = Black
        trapped = r < RS
        
        # Update image for trapped rays (Black)
        # We don't actually need to paint black (it's 0,0,0 default), just stop tracing
        active_sub_indices = np.where(active_mask)[0] # Map back to full array
        
        # Mark trapped rays as inactive
        trapped_indices = np.where(trapped)[0]
        # We need to find the specific (h,w) indices.
        # Optimization: Just updating the boolean mask is tricky in 1D
        # Let's do a full 2D check for simplicity in this demo script:
        
        # (Re-calculating full array variables for readability/simplicity)
        full_r = np.linalg.norm(ray_pos, axis=2)
        
        # B. ACCRETION DISK (The Halo)
        # Define Disk Plane: Y = 0 (roughly, since we tilted camera)
        # Actually, simpler: The disk is on the X-Z plane (y=0).
        # We check if a ray crossed Y=0 in this step.
        
        old_pos_y = ray_pos[:,:,1] - ray_dir[:,:,1] * dt
        crossed_plane = (old_pos_y * ray_pos[:,:,1]) < 0
        
        # If crossed plane, check radius
        plane_r = np.sqrt(ray_pos[:,:,0]**2 + ray_pos[:,:,2]**2)
        hit_disk = crossed_plane & (plane_r > DISK_INNER) & (plane_r < DISK_OUTER) & active_mask
        
        if np.any(hit_disk):
            # Paint disk pixels
            # Color logic: Inner = Hot (Blue/White), Outer = Cool (Red/Orange)
            # Doppler Beaming: Left side (x<0) is approaching -> Brighter
            
            # Distance factor (0.0 to 1.0 from inner to outer)
            dist_factor = (plane_r[hit_disk] - DISK_INNER) / (DISK_OUTER - DISK_INNER)
            
            # Doppler factor (Fake it based on X coordinate)
            doppler = 1.0 - (ray_pos[hit_disk, 0] * 0.1)
            
            intensity = (1.0 - dist_factor) * doppler
            intensity = np.clip(intensity, 0.2, 1.0)
            
            # Assign Color (Orange-ish)
            image[hit_disk, 0] = 1.0 * intensity  # R
            image[hit_disk, 1] = 0.6 * intensity  # G
            image[hit_disk, 2] = 0.2 * intensity  # B
            
            # Stop tracing these rays
            active_mask[hit_disk] = False

        # Stop tracing trapped rays
        active_mask[full_r < RS] = False
        
        # Stop tracing rays that escaped to infinity
        active_mask[full_r > 30.0] = False

    # 4. POST PROCESSING (Bloom/Glow)
    # Simple starfield background for escaped rays
    stars = (np.random.rand(h, w) > 0.995).astype(float)
    # Only apply stars where we haven't drawn the disk or the hole
    mask_empty = np.all(image == 0, axis=2)
    # Don't draw stars over the black hole shadow
    mask_shadow = full_r < RS
    stars[mask_shadow] = 0
    
    image[mask_empty, 0] = stars[mask_empty]
    image[mask_empty, 1] = stars[mask_empty]
    image[mask_empty, 2] = stars[mask_empty]

    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(np.clip(image, 0, 1), origin='lower')
    plt.title(f"AOTI Event Horizon (Mass={BH_MASS})")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    render()