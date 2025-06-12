import os
import glob
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from PIL import Image, ImageDraw
import gpxpy
import fitparse
import imageio
import numpy as np
import shutil
import time

# --- Configuration ---
# Directory containing your GPX or FIT files
TRACKS_DIR = "data"
# Directory to save temporary frames
FRAME_DIR = "frames"
# Output video file name
OUTPUT_VIDEO_FILE = "tracks_timelapse.mp4"
# Desired video duration in seconds
VIDEO_DURATION_S = 15
# Frames per second for the output video
FPS = 30
# Image dimensions for the video frames
IMG_WIDTH = 1024
IMG_HEIGHT = 1024
# Background color for the canvas
BG_COLOR = "white"
# Padding around the tracks in pixels
PADDING = 100
# Line width for drawing tracks
LINE_WIDTH = 5
# Colors to cycle through for different tracks
TRACK_COLORS = [
    (255, 69, 0),    # Red-Orange
    (23, 190, 207),  # Blue
    (44, 160, 44),   # Green
    (255, 127, 14),  # Orange
    (148, 103, 189), # Purple
    (227, 119, 194), # Pink
]

# --- Parsing Functions ---

def parse_gpx_file(file_path):
    """Parses a single GPX file, extracting track points."""
    try:
        with open(file_path, 'r', encoding='utf-8') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            points = [
                (point.latitude, point.longitude)
                for track in gpx.tracks
                for segment in track.segments
                for point in segment.points
            ]
            if not points:
                print(f"Warning: No points found in GPX file {os.path.basename(file_path)}")
            return points
    except Exception as e:
        print(f"Error parsing GPX {os.path.basename(file_path)}: {e}")
        return None

def parse_fit_file(file_path):
    """Parses a single FIT file, extracting track points."""
    try:
        fitfile = fitparse.FitFile(file_path)
        points = []
        # Semicircles to degrees conversion factor
        SEMI_TO_DEG = 180.0 / 2**31
        
        for record in fitfile.get_messages("record"):
            lat = record.get_value("position_lat")
            lon = record.get_value("position_long")
            if lat is not None and lon is not None:
                points.append((lat * SEMI_TO_DEG, lon * SEMI_TO_DEG))
        
        if not points:
            print(f"Warning: No points found in FIT file {os.path.basename(file_path)}")
        return points
    except Exception as e:
        print(f"Error parsing FIT {os.path.basename(file_path)}: {e}")
        return None

def parse_file_wrapper(file_path):
    """A wrapper to select the correct parser based on file extension."""
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.gpx':
        return parse_gpx_file(file_path)
    elif ext.lower() == '.fit':
        return parse_fit_file(file_path)
    return None

def get_all_tracks_parallel(directory):
    """
    Finds and parses all GPX and FIT files in a directory using a
    multiprocessing pool to significantly speed up the process.
    """
    supported_extensions = ("*.gpx", "*.fit")
    all_files = []
    for ext in supported_extensions:
        all_files.extend(glob.glob(os.path.join(directory, ext)))

    if not all_files:
        print(f"Error: No .gpx or .fit files found in '{directory}'.")
        return []

    tracks = []
    # Using a ProcessPoolExecutor to parallelize the file parsing,
    # which is a major bottleneck for large numbers of files.
    with ProcessPoolExecutor() as executor:
        # map() applies the parse_file_wrapper function to every file path.
        # The results are returned as an iterator as they complete.
        results = executor.map(parse_file_wrapper, all_files)
        # Filter out any files that failed to parse (returned None)
        tracks = [track for track in results if track]
            
    return tracks

# --- Core Data Processing ---

def normalize_tracks(tracks, num_points):
    """Normalizes all tracks to have the same number of points via interpolation."""
    normalized_tracks = []
    for track_points in tracks:
        track_np = np.array(track_points)
        current_indices = np.linspace(0, 1, len(track_np))
        target_indices = np.linspace(0, 1, num_points)
        
        interp_lat = np.interp(target_indices, current_indices, track_np[:, 0])
        interp_lon = np.interp(target_indices, current_indices, track_np[:, 1])
        
        normalized_tracks.append(list(zip(interp_lat, interp_lon)))
    return normalized_tracks

def project_points_to_pixels(tracks, width, height, padding):
    """Projects geographical coordinates to pixel coordinates to fit the image."""
    # Combine all points from all tracks to find the global bounding box
    all_points = np.vstack(tracks)
    
    min_lat, min_lon = all_points.min(axis=0)
    max_lat, max_lon = all_points.max(axis=0)

    lat_range = max_lat - min_lat
    lon_range = max_lon - min_lon

    if lat_range == 0 or lon_range == 0:
        scale = 1
    else:
        scale_x = (width - 2 * padding) / lon_range
        scale_y = (height - 2 * padding) / lat_range
        scale = min(scale_x, scale_y)

    projected_tracks = []
    for track in tracks:
        projected_track = [
            (padding + (lon - min_lon) * scale, padding + (max_lat - lat) * scale)
            for lat, lon in track
        ]
        projected_tracks.append(projected_track)
        
    return projected_tracks

# --- Video Generation ---

def generate_frame(frame_index, total_frames, tracks, colors, frame_dir):
    """Generates a single image frame for the animation."""
    points_to_draw = int((frame_index / (total_frames - 1)) * len(tracks[0]))
    # Ensure at least two points to draw a line segment
    if points_to_draw < 2:
        points_to_draw = 2

    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    for i, track_points in enumerate(tracks):
        color = colors[i % len(colors)]
        segment = track_points[:points_to_draw]
        draw.line(segment, fill=color, width=LINE_WIDTH, joint="curve")

    frame_path = os.path.join(frame_dir, f"frame_{frame_index:05d}.png")
    img.save(frame_path)
    return frame_path

# --- Main Execution ---

def main():
    """Main function to orchestrate the video generation process."""
    print("--- Fully Parallelized Track to Video Renderer ---")
    script_start_time = time.time()

    # 1. Setup directories
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR)
    
    if not os.path.exists(TRACKS_DIR):
        os.makedirs(TRACKS_DIR)
        print(f"Created directory '{TRACKS_DIR}'. Please add your .gpx/.fit files and run again.")
        return

    # 2. Parallel Parsing of all track files
    print("\n[Step 1/5] Parsing track files in parallel...")
    parse_start_time = time.time()
    tracks = get_all_tracks_parallel(TRACKS_DIR)
    if not tracks:
        print("Stopping: No valid tracks were loaded.")
        return
    parse_duration = time.time() - parse_start_time
    print(f"-> Parsed {len(tracks)} files in {parse_duration:.2f} seconds.")

    # 3. Normalization and Projection
    print("\n[Step 2/5] Normalizing and projecting tracks...")
    # This number determines the smoothness of the animation.
    num_normalized_points = (VIDEO_DURATION_S * FPS) * 4
    normalized_tracks = normalize_tracks(tracks, num_normalized_points)
    projected_tracks = project_points_to_pixels(normalized_tracks, IMG_WIDTH, IMG_HEIGHT, PADDING)
    print("-> Tracks are ready for rendering.")

    # 4. Parallel Frame Generation
    total_frames = VIDEO_DURATION_S * FPS
    print(f"\n[Step 3/5] Generating {total_frames} frames in parallel...")
    frame_gen_start_time = time.time()

    task = partial(
        generate_frame,
        total_frames=total_frames,
        tracks=projected_tracks,
        colors=TRACK_COLORS,
        frame_dir=FRAME_DIR,
    )
    
    with ProcessPoolExecutor() as executor:
        # We don't need to store the paths here, just execute the tasks
        list(executor.map(task, range(total_frames)))

    frame_gen_duration = time.time() - frame_gen_start_time
    print(f"\n-> Generated {total_frames} frames in {frame_gen_duration:.2f} seconds.")

    # 5. Video Assembly
    print("\n[Step 4/5] Assembling frames into video...")
    video_start_time = time.time()
    # Read the generated frame paths in sorted order
    frame_paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.png")))

    with imageio.get_writer(OUTPUT_VIDEO_FILE, fps=FPS, quality=8, codec='libx264') as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    video_duration = time.time() - video_start_time
    print(f"-> Video saved as '{OUTPUT_VIDEO_FILE}' in {video_duration:.2f} seconds.")

    # 6. Cleanup
    print("\n[Step 5/5] Cleaning up temporary frame files...")
    shutil.rmtree(FRAME_DIR)

    total_duration = time.time() - script_start_time
    print(f"\n--- Process Complete in {total_duration:.2f} seconds ---")

if __name__ == "__main__":
    main()
