import os
import glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import imageio.v2 as imageio
import shutil
import time
import gpxpy
import fitparse
import math
from io import BytesIO # Moved to top level to be accessible by all processes

# --- Renderer-specific imports - required for the script to run ---
# Pillow is used for CPU rendering
from PIL import Image, ImageDraw
# Pygame and PyOpenGL are used for GPU rendering
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
# Matplotlib is used for generating a large, visually distinct color palette
import matplotlib.cm as cm

# --- Imports for map background feature ---
import requests
import mercantile

# ==============================================================================
# --- Master Configuration ---
# ==============================================================================
class Config:
    # --- Core Settings ---
    # Choose your renderer: 'GPU' (requires PyOpenGL/Pygame) or 'CPU' (requires Pillow)
    RENDERER = 'GPU'  # <-- CHANGE THIS: 'GPU' or 'CPU'
    # Use multiple CPU cores for parsing/rendering. Set to False for single-threaded operation.
    USE_MULTIPROCESSING = True

    # --- Map Background Settings ---
    USE_MAP_BACKGROUND = False  # <-- SET TO True TO ENABLE MAPS
    # The zoom level for the map. 12-15 is good for city/regional level. Higher is more zoomed in.
    MAP_ZOOM = 13
    # Tile server URL. {z}/{x}/{y} are placeholders.
    TILE_SERVER_URL = "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"
    # A user-agent is polite and sometimes required by tile servers.
    HTTP_USER_AGENT = "GpxVideoRenderer/1.0 (https://your-github-or-website)"

    # --- File and Directory Settings ---
    TRACKS_DIR = "data"
    FRAME_DIR = "frames_temp"
    OUTPUT_VIDEO_FILE = "tracks_timelapse.mp4"
    # Path to save the stitched map background
    MAP_CACHE_FILE = "map_background.png"

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

    # --- Color Generation Settings ---
    # Number of unique colors to generate.
    NUM_COLORS = 20
    # Colormap to use. 'viridis', 'plasma', 'tab20', 'hsv' are good choices.
    COLORMAP = 'viridis'

    # --- Auto-Generated Colors ---
    try:
        # Generate N visually distinct colors using a matplotlib colormap
        _colormap = cm.get_cmap(COLORMAP, NUM_COLORS)
        # Get colors as floats (0-1) and take only RGB, discard Alpha
        _colors_float = _colormap(np.linspace(0, 1, NUM_COLORS))[:, :3]
        # Convert to the correct format for each renderer
        TRACK_COLORS_GPU = _colors_float.astype(np.float32)
        TRACK_COLORS_CPU = [tuple(int(c * 255) for c in color) for color in _colors_float]
    except Exception as e:
        print(f"Warning: Could not generate colors with Matplotlib ({e}). Falling back to default.")
        TRACK_COLORS_CPU = [(255, 69, 0), (23, 190, 207), (44, 160, 44)]
        TRACK_COLORS_GPU = np.array(TRACK_COLORS_CPU, dtype=np.float32) / 255.0

# ==============================================================================
# --- Map Generation ---
# ==============================================================================

def fetch_tile(tile_url):
    """Downloads a single map tile."""
    try:
        response = requests.get(tile_url, headers={'User-Agent': Config.HTTP_USER_AGENT})
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.RequestException as e:
        print(f"Warning: Failed to fetch tile {tile_url}. {e}")
        return Image.new('RGB', (256, 256), 'grey')

def create_map_background(tracks):
    """Calculates bounds, fetches, and stitches map tiles into a single image."""
    print("  Creating map background...")
    # Calculate the geographic bounding box for all tracks
    all_points = np.vstack([np.array(t) for t in tracks if t])
    west, south = all_points.min(axis=0)
    east, north = all_points.max(axis=0)
    
    # Get the list of tiles that cover this bounding box from mercantile
    tiles = list(mercantile.tiles(west, south, east, north, Config.MAP_ZOOM))
    
    min_x = min(t.x for t in tiles)
    max_x = max(t.x for t in tiles)
    min_y = min(t.y for t in tiles)
    max_y = max(t.y for t in tiles)

    map_width = (max_x - min_x + 1) * 256
    map_height = (max_y - min_y + 1) * 256
    
    background = Image.new('RGB', (map_width, map_height))
    
    tile_urls = [Config.TILE_SERVER_URL.format(z=t.z, x=t.x, y=t.y) for t in tiles]

    # Download tiles in parallel
    with ProcessPoolExecutor() as executor:
        tile_images = list(executor.map(fetch_tile, tile_urls))

    # Stitch tiles together
    for tile_obj, tile_img in zip(tiles, tile_images):
        x = (tile_obj.x - min_x) * 256
        y = (tile_obj.y - min_y) * 256
        background.paste(tile_img, (x, y))
        
    # We need the top-left geographic corner of our map to align the tracks
    top_left_bounds = mercantile.bounds(min_x, min_y, Config.MAP_ZOOM)
    background.save(Config.MAP_CACHE_FILE)
    print(f"  -> Map background saved to {Config.MAP_CACHE_FILE}")
    return top_left_bounds

# ==============================================================================
# --- Data Processing ---
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

def normalize_and_project_tracks(tracks, map_bounds=None):
    """Normalizes and projects tracks, either to a blank canvas or a map."""
    # Normalize tracks to have the same number of points for smooth animation
    num_normalized_points = (Config.VIDEO_DURATION_S * Config.FPS) * 4
    normalized_tracks = []
    for track_points in tracks:
        # Added a check for empty tracks
        if not track_points:
            continue
        track_np = np.array(track_points)
        current_indices = np.linspace(0, 1, len(track_np))
        target_indices = np.linspace(0, 1, num_normalized_points)
        interp_lat = np.interp(target_indices, current_indices, track_np[:, 0])
        interp_lon = np.interp(target_indices, current_indices, track_np[:, 1])
        normalized_tracks.append(np.column_stack((interp_lat, interp_lon)))

    if not normalized_tracks:
        return []

    projected_tracks = []
    if Config.USE_MAP_BACKGROUND and map_bounds:
        # --- Map Projection Logic ---
        # Convert lat/lon to pixel coordinates on the world map
        def latlon_to_pixels(lat, lon, zoom):
            lat_rad = np.radians(lat)
            n = 2.0 ** zoom
            xtile = (lon + 180.0) / 360.0 * n
            ytile = (1.0 - np.asinh(np.tan(lat_rad)) / math.pi) / 2.0 * n
            return xtile * 256, ytile * 256

        # Get pixel coords of the top-left corner of our stitched map
        map_origin_x, map_origin_y = latlon_to_pixels(map_bounds.north, map_bounds.west, Config.MAP_ZOOM)
        
        for track in normalized_tracks:
            px, py = latlon_to_pixels(track[:, 0], track[:, 1], Config.MAP_ZOOM)
            # Offset points to be relative to our stitched map image
            x = px - map_origin_x
            y = py - map_origin_y
            projected_tracks.append(np.column_stack((x, y)))
    else:
        # --- White Background Projection Logic ---
        all_points = np.vstack(normalized_tracks)
        min_lat, min_lon = all_points.min(axis=0); max_lat, max_lon = all_points.max(axis=0)
        lat_range, lon_range = max_lat - min_lat, max_lon - min_lon
        scale = 1
        if lat_range > 0 and lon_range > 0:
            scale = min((Config.IMG_WIDTH - 2*Config.PADDING)/lon_range, (Config.IMG_HEIGHT - 2*Config.PADDING)/lat_range)
        for track in normalized_tracks:
            x = Config.PADDING + (track[:, 1] - min_lon) * scale
            y = Config.PADDING + (max_lat - track[:, 0]) * scale
            projected_tracks.append(np.column_stack((x, y)))

    return projected_tracks

# ==============================================================================
# --- CPU (Pillow) Renderer ---
# ==============================================================================

# --- CPU Renderer ---
cpu_worker_tracks, cpu_worker_bg = None, None

def init_worker_cpu(tracks, bg_image=None):
    """Initializer for CPU worker processes."""
    global cpu_worker_tracks, cpu_worker_bg
    cpu_worker_tracks, cpu_worker_bg = tracks, bg_image

def generate_frame_cpu(frame_info):
    """Generates a single frame using Pillow (CPU)."""
    frame_index, total_frames, frame_dir = frame_info

    img = Image.new("RGB", (Config.IMG_WIDTH, Config.IMG_HEIGHT), Config.BG_COLOR_CPU)
    if cpu_worker_bg: img.paste(cpu_worker_bg, (0, 0)) # Paste map background
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
gpu_worker_vbo, gpu_worker_track_offsets, gpu_worker_track_lengths, gpu_worker_map_tex = None, None, None, None
def init_worker_gpu(all_tracks_flat, offsets, lengths, map_cache_file=None):
    """Initializes a Pygame/OpenGL context and uploads track data to the GPU."""
    global gpu_worker_vbo, gpu_worker_track_offsets, gpu_worker_track_lengths, gpu_worker_map_tex
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

    if map_cache_file and os.path.exists(map_cache_file):
        # Load map image and create an OpenGL texture
        map_surface = pygame.image.load(map_cache_file)
        map_data = pygame.image.tostring(map_surface, "RGB", 1)
        width, height = map_surface.get_width(), map_surface.get_height()

        gpu_worker_map_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, gpu_worker_map_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, map_data)

def generate_frame_gpu(frame_info):
    """Generates a single frame using the pre-initialized OpenGL context."""
    frame_index, total_frames, frame_dir = frame_info
    glClearColor(*Config.BG_COLOR_GPU)
    glClear(GL_COLOR_BUFFER_BIT)

    # Draw map background if texture exists
    if gpu_worker_map_tex:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, gpu_worker_map_tex)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(0, Config.IMG_HEIGHT)
        glTexCoord2f(1, 0); glVertex2f(Config.IMG_WIDTH, Config.IMG_HEIGHT)
        glTexCoord2f(1, 1); glVertex2f(Config.IMG_WIDTH, 0)
        glTexCoord2f(0, 1); glVertex2f(0, 0)
        glEnd()
        glDisable(GL_TEXTURE_2D)

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

    pygame.display.flip()
    glFinish()

    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, Config.IMG_WIDTH, Config.IMG_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
    frame_path = os.path.join(frame_dir, f"frame_{frame_index:05d}.png")
    imageio.imwrite(frame_path, np.frombuffer(pixels, dtype=np.uint8).reshape(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3))#[::-1, :, :])
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
    print("\n[Step 1/5] Parsing track files...")
    parse_start = time.time()
    tracks = get_tracks(Config.TRACKS_DIR, Config.USE_MULTIPROCESSING)
    if not tracks:
        print("Stopping: No valid tracks loaded."); return
    print(f"-> Parsed {len(tracks)} files in {time.time() - parse_start:.2f}s.")
    
    # --- Step 2: Map Generation ---
    map_bounds = None
    if Config.USE_MAP_BACKGROUND:
        print("\n[Step 2/5] Generating map background...")
        map_bounds = create_map_background(tracks)
    else: print("\n[Step 2/5] Skipping map background generation.")
        
    # --- Step 3: Data Preparation ---
    print("\n[Step 3/5] Normalizing and projecting track data...")
    projected_tracks = normalize_and_project_tracks(tracks, map_bounds)
    if not projected_tracks:
        print("Stopping: No data to render after normalization."); return
    print(f"-> Data prepared for rendering.")

    # --- Step 4: Frame Generation ---
    total_frames = Config.VIDEO_DURATION_S * Config.FPS
    print(f"\n[Step 4/5] Generating {total_frames} frames using {Config.RENDERER}...")
    frame_gen_start = time.time()
    frame_args = [(i, total_frames, Config.FRAME_DIR) for i in range(total_frames)]

    # Select the correct functions and arguments based on the chosen renderer
    if Config.RENDERER == 'GPU':
        init_worker_func = init_worker_gpu
        generate_frame_func = generate_frame_gpu
        flat_data_list, track_offsets, track_lengths, current_offset = [], [], [], 0
        for track in projected_tracks:
            projected = track.astype(np.float32)
            flat_data_list.append(projected)
            track_offsets.append(current_offset)
            track_lengths.append(len(projected))
            current_offset += len(projected)
        
        # Add a safety check in case all tracks were empty/invalid
        if not flat_data_list:
            print("Stopping: No data to send to GPU after processing."); return

        init_args = (np.vstack(flat_data_list), track_offsets, track_lengths, Config.MAP_CACHE_FILE if Config.USE_MAP_BACKGROUND else None)
    else: # Default to CPU
        init_worker_func = init_worker_cpu
        generate_frame_func = generate_frame_cpu
        bg_image = Image.open(Config.MAP_CACHE_FILE) if Config.USE_MAP_BACKGROUND and os.path.exists(Config.MAP_CACHE_FILE) else None
        init_args = (projected_tracks, bg_image)

    # Execute rendering, either in parallel or sequentially
    if Config.USE_MULTIPROCESSING:
        with ProcessPoolExecutor(initializer=init_worker_func, initargs=init_args) as executor:
            list(executor.map(generate_frame_func, frame_args))
    else:
        init_worker_func(*init_args)
        for arg in frame_args:
            generate_frame_func(arg)
            
    print(f"-> Generated {total_frames} frames in {time.time() - frame_gen_start:.2f}s.")
    # --- Step 5: Video Assembly ---
    print("\n[Step 5/5] Assembling video...")
    frame_paths = sorted(glob.glob(os.path.join(Config.FRAME_DIR, "*.png")))
    with imageio.get_writer(Config.OUTPUT_VIDEO_FILE, fps=Config.FPS, quality=8, codec='libx264') as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    shutil.rmtree(Config.FRAME_DIR)
    if Config.USE_MAP_BACKGROUND and os.path.exists(Config.MAP_CACHE_FILE): os.remove(Config.MAP_CACHE_FILE);
    print(f"\n--- Process Complete in {time.time() - script_start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()
