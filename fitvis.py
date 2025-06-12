import os
import glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import imageio.v2 as imageio
import shutil
import time
import gpxpy
import fitparse

# --- Renderer-specific imports - required for the script to run ---
# Pillow is used for CPU rendering
from PIL import Image, ImageDraw
# Pygame and PyOpenGL are used for GPU rendering
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# ==============================================================================
# --- Master Configuration ---
# ==============================================================================
class Config:
    # --- Core Settings ---
    # Choose your renderer: 'GPU' (requires PyOpenGL/Pygame) or 'CPU' (requires Pillow)
    RENDERER = 'GPU'  # <-- CHANGE THIS: 'GPU' or 'CPU'
    # Use multiple CPU cores for parsing/rendering. Set to False for single-threaded operation.
    USE_MULTIPROCESSING = True

    # --- File and Directory Settings ---
    TRACKS_DIR = "data"
    FRAME_DIR = "frames_temp"
    OUTPUT_VIDEO_FILE = "tracks_timelapse.mp4"

    # --- Video Settings ---
    VIDEO_DURATION_S = 15
    FPS = 30

    # --- Visual Settings ---
    IMG_WIDTH = 1024
    IMG_HEIGHT = 1024
    PADDING = 10
    LINE_WIDTH = 5
    # Background color (don't change format)
    BG_COLOR_CPU = "white" # For Pillow
    BG_COLOR_GPU = (1.0, 1.0, 1.0, 1.0) # For OpenGL (RGBA, 0-1)
    # Track colors (don't change format)
    TRACK_COLORS_CPU = [(255, 69, 0), (23, 190, 207), (44, 160, 44), (255, 127, 14), (148, 103, 189), (227, 119, 194)]
    TRACK_COLORS_GPU = np.array(TRACK_COLORS_CPU, dtype=np.float32) / 255.0

# ==============================================================================
# --- Parsing and Data Processing (Common to both renderers) ---
# ==============================================================================

def parse_file_wrapper(file_path):
    """Selects the correct parser based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.gpx':
            with open(file_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
                return [(p.latitude, p.longitude) for t in gpx.tracks for s in t.segments for p in s.points]
        elif ext == '.fit':
            fit = fitparse.FitFile(file_path)
            SEMI_TO_DEG = 180.0 / 2**31
            points = []
            for r in fit.get_messages("record"):
                lat, lon = r.get_value("position_lat"), r.get_value("position_long")
                if lat is not None and lon is not None:
                    points.append((lat * SEMI_TO_DEG, lon * SEMI_TO_DEG))
            return points
    except Exception:
        return None
    return None

def get_tracks(directory, parallel):
    """Finds and parses all files, using parallel or sequential logic."""
    files = glob.glob(os.path.join(directory, "*.gpx")) + glob.glob(os.path.join(directory, "*.fit"))
    if not files: return []

    if parallel:
        with ProcessPoolExecutor() as executor:
            tracks = list(executor.map(parse_file_wrapper, files))
    else:
        tracks = [parse_file_wrapper(f) for f in files]
    
    return list(filter(None, tracks)) # Remove failed parses

def normalize_and_project_tracks(tracks):
    """Converts raw track data into normalized, projected pixel coordinates."""
    # Normalize tracks to have the same number of points for smooth animation
    num_normalized_points = (Config.VIDEO_DURATION_S * Config.FPS) * 4
    normalized_tracks = []
    for track_points in tracks:
        track_np = np.array(track_points)
        current_indices = np.linspace(0, 1, len(track_np))
        target_indices = np.linspace(0, 1, num_normalized_points)
        interp_lat = np.interp(target_indices, current_indices, track_np[:, 0])
        interp_lon = np.interp(target_indices, current_indices, track_np[:, 1])
        normalized_tracks.append(np.column_stack((interp_lat, interp_lon)))

    # Project geographical coordinates to pixel space
    all_points = np.vstack(normalized_tracks)
    min_lat, min_lon = all_points.min(axis=0); max_lat, max_lon = all_points.max(axis=0)
    lat_range, lon_range = max_lat - min_lat, max_lon - min_lon
    scale = 1
    if lat_range > 0 and lon_range > 0:
        scale = min((Config.IMG_WIDTH - 2*Config.PADDING)/lon_range, (Config.IMG_HEIGHT - 2*Config.PADDING)/lat_range)
    
    projected_tracks = []
    for track in normalized_tracks:
        x = Config.PADDING + (track[:, 1] - min_lon) * scale
        y = Config.PADDING + (max_lat - track[:, 0]) * scale
        projected_tracks.append(np.column_stack((x, y)))
        
    return projected_tracks

# ==============================================================================
# --- CPU (Pillow) Renderer ---
# ==============================================================================

# Globals for the CPU worker processes
cpu_worker_tracks = None

def init_worker_cpu(tracks):
    """Initializer for CPU worker processes."""
    global cpu_worker_tracks
    cpu_worker_tracks = tracks

def generate_frame_cpu(frame_info):
    """Generates a single frame using Pillow (CPU)."""
    frame_index, total_frames, frame_dir = frame_info
    
    img = Image.new("RGB", (Config.IMG_WIDTH, Config.IMG_HEIGHT), Config.BG_COLOR_CPU)
    draw = ImageDraw.Draw(img)
    
    points_to_draw = int((frame_index / (total_frames - 1)) * len(cpu_worker_tracks[0]))
    if points_to_draw < 2: points_to_draw = 2

    for i, track_points_np in enumerate(cpu_worker_tracks):
        color = Config.TRACK_COLORS_CPU[i % len(Config.TRACK_COLORS_CPU)]
        segment = track_points_np[:points_to_draw]
        # Pillow's line method needs a list of tuples
        draw.line(list(map(tuple, segment)), fill=color, width=Config.LINE_WIDTH, joint="curve")
        
    frame_path = os.path.join(frame_dir, f"frame_{frame_index:05d}.png")
    img.save(frame_path)
    return frame_path

# ==============================================================================
# --- GPU (OpenGL) Renderer ---
# ==============================================================================

# Globals for the GPU worker processes
gpu_worker_vbo, gpu_worker_track_offsets, gpu_worker_track_lengths = None, None, None

def init_worker_gpu(all_tracks_flat, offsets, lengths):
    """Initializes a Pygame/OpenGL context and uploads track data to the GPU."""
    global gpu_worker_vbo, gpu_worker_track_offsets, gpu_worker_track_lengths
    pygame.init()
    pygame.display.set_mode((Config.IMG_WIDTH, Config.IMG_HEIGHT), DOUBLEBUF | OPENGL | HIDDEN)
    glViewport(0, 0, Config.IMG_WIDTH, Config.IMG_HEIGHT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, Config.IMG_WIDTH, 0, Config.IMG_HEIGHT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glEnable(GL_LINE_SMOOTH); glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glLineWidth(Config.LINE_WIDTH)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, all_tracks_flat.nbytes, all_tracks_flat, GL_STATIC_DRAW)
    gpu_worker_vbo, gpu_worker_track_offsets, gpu_worker_track_lengths = vbo, offsets, lengths

def generate_frame_gpu(frame_info):
    """Generates a single frame using the pre-initialized OpenGL context."""
    frame_index, total_frames, frame_dir = frame_info
    glClearColor(*Config.BG_COLOR_GPU)
    glClear(GL_COLOR_BUFFER_BIT)
    glEnableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, gpu_worker_vbo)
    glVertexPointer(2, GL_FLOAT, 0, None)

    points_to_draw = int((frame_index / (total_frames - 1)) * gpu_worker_track_lengths[0])
    if points_to_draw < 2: points_to_draw = 2
    
    for i in range(len(gpu_worker_track_offsets)):
        glColor3fv(Config.TRACK_COLORS_GPU[i % len(Config.TRACK_COLORS_GPU)])
        start_index = gpu_worker_track_offsets[i]
        glDrawArrays(GL_LINE_STRIP, start_index, min(points_to_draw, gpu_worker_track_lengths[i]))

    glDisableClientState(GL_VERTEX_ARRAY)
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, Config.IMG_WIDTH, Config.IMG_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
    frame_path = os.path.join(frame_dir, f"frame_{frame_index:05d}.png")
    imageio.imwrite(frame_path, np.frombuffer(pixels, dtype=np.uint8).reshape(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3)[::-1, :, :])
    return frame_path

# ==============================================================================
# --- Main Execution ---
# ==============================================================================

def main():
    print(f"--- Unified Track Renderer ---")
    print(f"MODE: {Config.RENDERER} | Multiprocessing: {'ENABLED' if Config.USE_MULTIPROCESSING else 'DISABLED'}")
    script_start_time = time.time()

    if os.path.exists(Config.FRAME_DIR): shutil.rmtree(Config.FRAME_DIR)
    os.makedirs(Config.FRAME_DIR)
    if not os.path.exists(Config.TRACKS_DIR): os.makedirs(Config.TRACKS_DIR)

    # --- Step 1: Parsing ---
    print("\n[Step 1/4] Parsing track files...")
    parse_start = time.time()
    tracks = get_tracks(Config.TRACKS_DIR, Config.USE_MULTIPROCESSING)
    if not tracks:
        print("Stopping: No valid tracks loaded."); return
    print(f"-> Parsed {len(tracks)} files in {time.time() - parse_start:.2f}s.")

    # --- Step 2: Data Preparation ---
    print("\n[Step 2/4] Normalizing and projecting track data...")
    projected_tracks = normalize_and_project_tracks(tracks)
    print(f"-> Data prepared for rendering.")

    # --- Step 3: Frame Generation ---
    total_frames = Config.VIDEO_DURATION_S * Config.FPS
    print(f"\n[Step 3/4] Generating {total_frames} frames using {Config.RENDERER}...")
    frame_gen_start = time.time()
    frame_args = [(i, total_frames, Config.FRAME_DIR) for i in range(total_frames)]

    # Select the correct functions and arguments based on the chosen renderer
    if Config.RENDERER == 'GPU':
        init_worker_func = init_worker_gpu
        generate_frame_func = generate_frame_gpu
        # Prepare data for VBO: flatten everything into one array
        flat_data_list, track_offsets, track_lengths, current_offset = [], [], [], 0
        for track in projected_tracks:
            projected = track.astype(np.float32)
            flat_data_list.append(projected)
            track_offsets.append(current_offset)
            track_lengths.append(len(projected))
            current_offset += len(projected)
        init_args = (np.vstack(flat_data_list), track_offsets, track_lengths)
    else: # Default to CPU
        init_worker_func = init_worker_cpu
        generate_frame_func = generate_frame_cpu
        init_args = (projected_tracks,)

    # Execute rendering, either in parallel or sequentially
    if Config.USE_MULTIPROCESSING:
        with ProcessPoolExecutor(initializer=init_worker_func, initargs=init_args) as executor:
            list(executor.map(generate_frame_func, frame_args))
    else:
        init_worker_func(*init_args)
        for arg in frame_args:
            generate_frame_func(arg)
            
    print(f"-> Generated {total_frames} frames in {time.time() - frame_gen_start:.2f}s.")

    # --- Step 4: Video Assembly ---
    print("\n[Step 4/4] Assembling video...")
    frame_paths = sorted(glob.glob(os.path.join(Config.FRAME_DIR, "*.png")))
    with imageio.get_writer(Config.OUTPUT_VIDEO_FILE, fps=Config.FPS, quality=8, codec='libx264') as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    shutil.rmtree(Config.FRAME_DIR)
    print(f"\n--- Process Complete in {time.time() - script_start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
