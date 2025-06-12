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
IMG_WIDTH = 1920
IMG_HEIGHT = 1080
# Background color for the canvas
BG_COLOR = "black"
# Padding around the tracks in pixels
PADDING = 100
# Line width for drawing tracks
LINE_WIDTH = 5
# Colors to cycle through for different tracks. Add more if you have many tracks.
TRACK_COLORS = [
    (255, 69, 0),    # Red-Orange
    (23, 190, 207),  # Blue
    (44, 160, 44),   # Green
    (255, 127, 14),  # Orange
    (148, 103, 189), # Purple
    (227, 119, 194), # Pink
]

# --- Parsing Logic ---

def parse_gpx_file(file_path):
    """Parses a GPX file and extracts all track points as a list of (lat, lon) tuples."""
    try:
        with open(file_path, 'r', encoding='utf-8') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            points = []
            for track in gpx.tracks:
                for segment in track.segments:
                    for point in segment.points:
                        points.append((point.latitude, point.longitude))
            print(f"Successfully parsed GPX {file_path} with {len(points)} points.")
            return points
    except Exception as e:
        print(f"Error parsing GPX file {file_path}: {e}")
        return None

def parse_fit_file(file_path):
    """Parses a FIT file and extracts all track points as a list of (lat, lon) tuples."""
    try:
        fitfile = fitparse.FitFile(file_path)
        points = []
        # Iterate through all data messages that are of type 'record'
        for record in fitfile.get_messages("record"):
            lat = record.get_value("position_lat")
            lon = record.get_value("position_long")
            
            # FIT files store lat/lon in 'semicircles', which need conversion to degrees.
            if lat is not None and lon is not None:
                lat_deg = lat * (180.0 / 2**31)
                lon_deg = lon * (180.0 / 2**31)
                points.append((lat_deg, lon_deg))

        print(f"Successfully parsed FIT {file_path} with {len(points)} points.")
        return points
    except Exception as e:
        print(f"Error parsing FIT file {file_path}: {e}")
        return None

def get_all_tracks(directory):
    """Finds all GPX and FIT files in a directory and parses them."""
    gpx_files = glob.glob(os.path.join(directory, "*.gpx"))
    fit_files = glob.glob(os.path.join(directory, "*.fit"))
    all_files = gpx_files + fit_files # Combine the lists

    if not all_files:
        print(f"Error: No .gpx or .fit files found in the '{directory}' directory.")
        return []
    
    tracks = []
    for file_path in all_files:
        # Determine which parser to use based on the file extension
        if file_path.lower().endswith('.gpx'):
            points = parse_gpx_file(file_path)
        elif file_path.lower().endswith('.fit'):
            points = parse_fit_file(file_path)
        else:
            points = None
        
        if points:
            tracks.append(points)
    return tracks

# --- Core Processing (Unchanged) ---

def normalize_tracks(tracks, num_points):
    """
    Normalizes all tracks to have the same number of points using interpolation.
    This is key to making them animate at the same rate.
    """
    normalized_tracks = []
    for track_points in tracks:
        track_np = np.array(track_points)
        current_num_points = len(track_np)
        current_indices = np.linspace(0, 1, current_num_points)
        target_indices = np.linspace(0, 1, num_points)
        interp_lat = np.interp(target_indices, current_indices, track_np[:, 0])
        interp_lon = np.interp(target_indices, current_indices, track_np[:, 1])
        normalized_tracks.append(list(zip(interp_lat, interp_lon)))
    return normalized_tracks

def project_points_to_pixels(tracks, width, height, padding):
    """
    Projects geographical coordinates to pixel coordinates, scaling them to fit the image.
    """
    all_points = np.array([point for track in tracks for point in track])
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
        projected_track = []
        for lat, lon in track:
            x = padding + (lon - min_lon) * scale
            y = padding + (max_lat - lat) * scale
            projected_track.append((x, y))
        projected_tracks.append(projected_track)
        
    return projected_tracks

def generate_frame(frame_index, total_frames, tracks, colors, frame_dir):
    """
    Generates a single image frame for the animation. This function is designed
    to be run in a separate process.
    """
    points_to_draw = int((frame_index / (total_frames - 1)) * len(tracks[0]))
    if points_to_draw < 2:
        points_to_draw = 2

    img = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    for i, track_points in enumerate(tracks):
        color = colors[i % len(colors)]
        segment_to_draw = track_points[:points_to_draw]
        if len(segment_to_draw) > 1:
            draw.line(segment_to_draw, fill=color, width=LINE_WIDTH, joint="curve")

    frame_path = os.path.join(frame_dir, f"frame_{frame_index:05d}.png")
    img.save(frame_path)
    return frame_path

def main():
    """Main function to orchestrate the video generation process."""
    print("--- GPX/FIT to Video Renderer ---")

    # 1. Setup directories
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR)
    
    if not os.path.exists(TRACKS_DIR):
        print(f"Error: The directory '{TRACKS_DIR}' does not exist.")
        print("Please create it and place your .gpx and .fit files inside.")
        return

    # 2. Load and parse all track files
    print("Step 1: Parsing GPX and FIT files...")
    tracks = get_all_tracks(TRACKS_DIR)
    if not tracks:
        print("Stopping because no tracks were loaded.")
        return

    # 3. Normalize tracks to have the same number of points
    print("Step 2: Normalizing track lengths for synchronized animation...")
    num_normalized_points = (VIDEO_DURATION_S * FPS) * 5 
    normalized_tracks = normalize_tracks(tracks, num_normalized_points)

    # 4. Project GPS coordinates to pixel coordinates
    print("Step 3: Projecting GPS coordinates to image space...")
    projected_tracks = project_points_to_pixels(normalized_tracks, IMG_WIDTH, IMG_HEIGHT, PADDING)

    # 5. Generate all frames using multiprocessing
    total_frames = VIDEO_DURATION_S * FPS
    print(f"Step 4: Generating {total_frames} frames using multiprocessing...")
    
    task = partial(
        generate_frame,
        total_frames=total_frames,
        tracks=projected_tracks,
        colors=TRACK_COLORS,
        frame_dir=FRAME_DIR,
    )
    
    frame_paths = []
    with ProcessPoolExecutor() as executor:
        for i, frame_path in enumerate(executor.map(task, range(total_frames))):
            frame_paths.append(frame_path)
            progress = (i + 1) / total_frames * 100
            print(f"  -> Generated frame {i + 1}/{total_frames} ({progress:.1f}%)", end="\r")

    print("\nAll frames generated successfully.")

    # 6. Create video from frames
    print("Step 5: Assembling frames into video...")
    with imageio.get_writer(OUTPUT_VIDEO_FILE, fps=FPS, quality=8) as writer:
        for frame_path in sorted(frame_paths):
            writer.append_data(imageio.imread(frame_path))

    print(f"Video saved as '{OUTPUT_VIDEO_FILE}'")

    # 7. Clean up temporary frame files
    print("Step 6: Cleaning up temporary frame files...")
    shutil.rmtree(FRAME_DIR)

    print("--- Process Complete ---")

if __name__ == "__main__":
    main()
