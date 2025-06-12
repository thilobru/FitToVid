# High-Performance GPX/FIT Track Video Renderer

## Overview

This project is a highly optimized Python script that visualizes multiple GPS tracks from `.gpx` or `.fit` files into a single, animated time-lapse video. It's designed to handle a large number of tracks efficiently by leveraging modern hardware capabilities, including multi-core CPUs and GPUs.

The final output is a video file showing all tracks being drawn simultaneously from their starting points, creating a beautiful "growing" visualization of the collective journeys.

## Features

* **Dual-Format Support:** Parses both `.gpx` and `.fit` files automatically.
* **Dual Rendering Engines:**
    * **CPU Renderer:** Uses the Pillow library for reliable, platform-agnostic image creation.
    * **GPU Renderer:** Uses OpenGL for massively accelerated frame generation, offloading the most intensive work to the graphics card.
* **Parallel Processing:** Leverages multiprocessing to drastically reduce processing time on multi-core systems for both file parsing and frame generation.
* **Highly Configurable:** A centralized `Config` class allows for easy changes to video dimensions, duration, colors, performance settings, and more.
* **Dynamic Color Palette:** Automatically generates a large, visually distinct set of colors using `matplotlib` colormaps, ensuring that even hundreds of tracks look unique.

## Setup & Usage

### Step 1: Install Dependencies

The script requires several Python libraries. You can install them all with `pip` using the following command:

```bash 
pip install numpy imageio gpxpy fitparse Pillow pygame PyOpenGL PyOpenGL_accelerate matplotlib
```

### Step 2: Organize Your Files

1. Create a folder named `data` in the same directory as the script. (You can change this name in the `Config` class).
2. Place all your `.gpx` and `.fit` files inside this `data` folder.

### Step 3: Configure the Script

Open the `main_unified.py` file and edit the `Config` class at the top to suit your needs. The most important settings are:

* `RENDERER`: Switch between `'GPU'` and `'CPU'`.
* `USE_MULTIPROCESSING`: Set to `True` for maximum speed on multi-core machines, or `False` to debug or run on a single core.
* `TRACKS_DIR`: The name of the folder containing your track files.
* `OUTPUT_VIDEO_FILE`: The desired name for the final video.
* `VIDEO_DURATION_S`, `FPS`: To control the length and smoothness of the video.

### Step 4: Run the Script

Execute the script from your terminal using this command:

```bash 
python main_unified.py
```

The script will print its progress through the steps and notify you when the video is complete. The final `.mp4` file will be saved in the main project directory.

## Performance & Architecture

The script is architected to overcome the common performance bottlenecks associated with this type of task.

### The Processing Pipeline

1. **Parsing:** The script first reads all track files. This can be time-consuming for hundreds of files, so it is parallelized across multiple CPU cores.
2. **Normalization & Projection:** All tracks are normalized to have the same number of points (to ensure they animate in sync) and are projected from GPS coordinates to pixel coordinates. This step is extremely fast thanks to the NumPy library.
3. **Frame Generation:** This is the most intensive step. Each frame of the video is generated as a separate image.
    * **CPU Mode:** The Pillow library draws the lines. This work is distributed across all CPU cores.
    * **GPU Mode:** All track data is uploaded to the GPU's memory once. The GPU then renders each frame with incredible speed. This work is still managed by processes on each CPU core, but the drawing commands themselves are executed by the GPU.
4. **Video Assembly:** The final step uses `imageio` to stitch all the generated frame images into a single `.mp4` video file.

### Complexity and Parallelism

* **Without Parallelism:** The total processing time would scale linearly with the number of files and the number of frames. For 100+ files, this would result in extremely long wait times.
* **With Parallelism:** By distributing the work across multiple CPU cores (C), the time complexity for the most demanding steps is roughly reduced by a factor of C. This makes the process viable for large datasets. The GPU renderer takes this a step further, reducing the frame generation time by another order of magnitude, often making parsing the main bottleneck once again.