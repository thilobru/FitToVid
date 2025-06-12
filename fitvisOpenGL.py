# main_gpu.py
import os
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import imageio
import shutil
import time

# --- Existing Parsing and Data Processing Functions (Unchanged) ---
# (These are kept from the previous script as they are efficient)
import gpxpy
import fitparse

def parse_gpx_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            return [(p.latitude, p.longitude) for t in gpx.tracks for s in t.segments for p in s.points]
    except Exception: return None

def parse_fit_file(file_path):
    try:
        fitfile = fitparse.FitFile(file_path)
        SEMI_TO_DEG = 180.0 / 2**31
        points = []
        for r in fitfile.get_messages("record"):
            lat, lon = r.get_value("position_lat"), r.get_value("position_long")
            if lat is not None and lon is not None:
                points.append((lat * SEMI_TO_DEG, lon * SEMI_TO_DEG))
        return points
    except Exception: return None

def parse_file_wrapper(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.gpx': return parse_gpx_file(file_path)
    if ext == '.fit': return parse_fit_file(file_path)
    return None

def get_all_tracks_parallel(directory):
    files = glob.glob(os.path.join(directory, "*.gpx")) + glob.glob(os.path.join(directory, "*.fit"))
    if not files: return []
    with ProcessPoolExecutor() as executor:
        tracks = list(filter(None, executor.map(parse_file_wrapper, files)))
    return tracks

def normalize_tracks(tracks, num_points):
    normalized = []
    for track_points in tracks:
        track_np = np.array(track_points)
        current_indices = np.linspace(0, 1, len(track_np))
        target_indices = np.linspace(0, 1, num_points)
        interp_lat = np.interp(target_indices, current_indices, track_np[:, 0])
        interp_lon = np.interp(target_indices, current_indices, track_np[:, 1])
        normalized.append(np.column_stack((interp_lat, interp_lon)))
    return normalized

# --- OpenGL Rendering Implementation ---

# Configuration (some settings moved here)
IMG_WIDTH, IMG_HEIGHT = 1024, 1024
BG_COLOR = (1.0, 1.0, 1.0, 1.0) # OpenGL uses 0-1 floats for RGBA
LINE_WIDTH = 5
TRACK_COLORS = np.array([
    [255, 69, 0], [23, 190, 207], [44, 160, 44],
    [255, 127, 14], [148, 103, 189], [227, 119, 194]
], dtype=np.float32) / 255.0 # Normalize colors to 0-1 for OpenGL

# --- Global variables for worker processes ---
# These will be initialized once per worker to avoid redundant setup
worker_vbos = None
worker_track_offsets = None
worker_track_lengths = None

def init_worker_gl(all_tracks_flat, offsets, lengths):
    """
    This function is run once per worker process. It initializes Pygame and OpenGL
    and uploads all track data to the GPU's memory (VBOs).
    """
    global worker_vbos, worker_track_offsets, worker_track_lengths

    # Create a hidden window for our OpenGL context
    pygame.init()
    pygame.display.set_mode((IMG_WIDTH, IMG_HEIGHT), DOUBLEBUF | OPENGL | HIDDEN)

    # Basic OpenGL setup
    glViewport(0, 0, IMG_WIDTH, IMG_HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, IMG_WIDTH, 0, IMG_HEIGHT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Enable line smoothing for anti-aliasing
    glEnable(GL_LINE_SMOOTH)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glLineWidth(LINE_WIDTH)

    # --- VBO (Vertex Buffer Object) setup ---
    # This is the key to performance: we send the data to the GPU ONCE.
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, all_tracks_flat.nbytes, all_tracks_flat, GL_STATIC_DRAW)
    
    # Store the VBO ID and track info in the worker's global scope
    worker_vbos = vbo
    worker_track_offsets = offsets
    worker_track_lengths = lengths

def generate_frame_gl(frame_info):
    """
    Generates a single frame using the pre-initialized OpenGL context.
    This function is lean and fast, only containing drawing commands.
    """
    frame_index, total_frames, frame_dir = frame_info

    # Clear the screen with the background color
    glClearColor(*BG_COLOR)
    glClear(GL_COLOR_BUFFER_BIT)

    # Enable vertex arrays to draw from our VBO
    glEnableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, worker_vbos)
    glVertexPointer(2, GL_FLOAT, 0, None)

    # Calculate how many points of each track to draw for this frame
    points_to_draw = int((frame_index / (total_frames - 1)) * worker_track_lengths[0])
    if points_to_draw < 2:
        points_to_draw = 2
    
    # Draw each track segment using the data already on the GPU
    for i in range(len(worker_track_offsets)):
        color = TRACK_COLORS[i % len(TRACK_COLORS)]
        glColor3fv(color)
        
        start_index = worker_track_offsets[i]
        glDrawArrays(GL_LINE_STRIP, start_index, min(points_to_draw, worker_track_lengths[i]))

    glDisableClientState(GL_VERTEX_ARRAY)

    # Read the pixels from the OpenGL buffer back to the CPU
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, IMG_WIDTH, IMG_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)

    # Save the frame using imageio
    frame_path = os.path.join(frame_dir, f"frame_{frame_index:05d}.png")
    # We need to flip the image vertically because OpenGL's origin is bottom-left
    imageio.imwrite(frame_path, np.frombuffer(pixels, dtype=np.uint8).reshape(IMG_HEIGHT, IMG_WIDTH, 3)[::-1, :, :])
    
    return frame_path

def main():
    """Main function to orchestrate the video generation process."""
    print("--- GPU-Accelerated Track to Video Renderer (OpenGL) ---")
    script_start_time = time.time()

    # --- Setup ---
    TRACKS_DIR = "data/100"
    FRAME_DIR = "frames"
    OUTPUT_VIDEO_FILE = "tracks_timelapse_gpu.mp4"
    VIDEO_DURATION_S, FPS = 15, 30
    PADDING = 100

    if os.path.exists(FRAME_DIR): shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR)
    if not os.path.exists(TRACKS_DIR): os.makedirs(TRACKS_DIR)

    # --- Step 1: Parallel Parsing (Same as before) ---
    print("\n[Step 1/4] Parsing track files...")
    tracks = get_all_tracks_parallel(TRACKS_DIR)
    if not tracks:
        print("Stopping: No valid tracks were loaded.")
        return
    print(f"-> Parsed {len(tracks)} files.")

    # --- Step 2: Data Preparation for GPU ---
    print("\n[Step 2/4] Normalizing and preparing data for GPU...")
    num_normalized_points = (VIDEO_DURATION_S * FPS) * 4
    normalized_tracks = normalize_tracks(tracks, num_normalized_points)
    
    # Project points to pixel coordinates
    all_points = np.vstack(normalized_tracks)
    min_lat, min_lon = all_points.min(axis=0)
    max_lat, max_lon = all_points.max(axis=0)
    lat_range, lon_range = max_lat - min_lat, max_lon - min_lon
    scale = min((IMG_WIDTH - 2 * PADDING) / lon_range, (IMG_HEIGHT - 2 * PADDING) / lat_range) if lat_range > 0 and lon_range > 0 else 1
    
    # Flatten all track data into one giant NumPy array for the VBO
    flat_data_list = []
    track_offsets = []
    track_lengths = []
    current_offset = 0
    for track in normalized_tracks:
        x = PADDING + (track[:, 1] - min_lon) * scale
        y = PADDING + (max_lat - track[:, 0]) * scale
        projected_track = np.column_stack((x, y)).astype(np.float32)
        
        flat_data_list.append(projected_track)
        track_offsets.append(current_offset)
        track_lengths.append(len(projected_track))
        current_offset += len(projected_track)
        
    all_tracks_flat = np.vstack(flat_data_list)
    print(f"-> Data prepared with {len(all_tracks_flat)} total vertices.")

    # --- Step 3: GPU-Accelerated Frame Generation ---
    total_frames = VIDEO_DURATION_S * FPS
    print(f"\n[Step 3/4] Generating {total_frames} frames using GPU...")
    frame_gen_start_time = time.time()
    
    frame_args = [(i, total_frames, FRAME_DIR) for i in range(total_frames)]

    with ProcessPoolExecutor(
        max_workers=os.cpu_count(),
        initializer=init_worker_gl,
        initargs=(all_tracks_flat, track_offsets, track_lengths)
    ) as executor:
        list(executor.map(generate_frame_gl, frame_args))

    frame_gen_duration = time.time() - frame_gen_start_time
    print(f"\n-> Generated {total_frames} frames in {frame_gen_duration:.2f} seconds.")

    # --- Step 4: Video Assembly ---
    print("\n[Step 4/4] Assembling video...")
    frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.png")))
    with imageio.get_writer(OUTPUT_VIDEO_FILE, fps=FPS, quality=8, codec='libx264') as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    # --- Cleanup ---
    shutil.rmtree(FRAME_DIR)
    total_duration = time.time() - script_start_time
    print(f"\n--- Process Complete in {total_duration:.2f} seconds ---")

if __name__ == "__main__":
    main()
