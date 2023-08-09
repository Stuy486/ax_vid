import numpy as np

#VIDEO DIMENSIONS
TIMING_HEIGHT = 50 # "Elapsed time: ##.##s (+/-#.###s)" just below the video
SLIP_BAR_HEIGHT = 50 # Top half is the bar, bottom half is the text
SLIP_TEXT_HEIGHT = 50 # Top half is the bar, bottom half is the text
TIME_SLIP_GRAPH_HEIGHT = (1080 - 540 - TIMING_HEIGHT - SLIP_BAR_HEIGHT - SLIP_TEXT_HEIGHT) # Calculated to make a 1080p video
COST_MATRIX_OVERLAY_SIZE = 300 # pixels

#ARGUMENTS
DATA_HZ = 30 # Frequency of cone comparison for alignment. Higher is better, but slower
#OUTPUT CONTROL
OUTPUT_ANNOTATED_VIDEOS = False # Output copies of the input videos, but with cones annotated
OUTPUT_TIME_SLIP_VIDEO = True # Output the side-by-side time slip video
OUTPUT_COST_MATRICES = True # Save cost matrices (raw, w/ dtw path, w/ smoothed poly path) as images
OUTPUT_FPS = 60 # Broken with USE_POLY_KEY_FRAME_SMOOTHING.
#TUNABLES
TARGET_DTW_AVERAGE_SCORE = 220 # 255 max, lower means more "gain" in cost matrix
COST_MATRIX_RECENTER_INTERVAL = 10 # seconds, controls how frequently we estimate time slip while calculating cost matrix
MAX_TIME_SLIP_PER_SECOND = .3 # In seconds. Increase if speed of one run is substantially different from the other. Larger value increases cost matrix creation time
USE_POLY_KEY_FRAME_SMOOTHING = True # Use polynomial smoothing when traversing path. Video is processed frame-by-frame rather than timestamp-to-timestamp, much faster
KEY_FRAME_COST_THRESHOLD = 230 # 255 max, higher means fewer but more confident key frames
MIN_KEY_FRAME_DELTA = 1.0 # seconds
MAX_KEY_FRAME_DELTA = 6.0 # seconds
TARGET_CONES_PER_FRAME = 8 # Target average number of cones per frame. Higher will take longer to generate cost matrix, but it will be more accurate.
                           # Most videos have at most 2 cones per frame availble.
#DEBUGGING
OVERLAY_COST_MATRIX_VISUALIZATION = False # Overlay the current snippet of the cost matrix in the output video
LOAD_CACHED_COST_MATRIX = False # load from file rather than recomputing
ONLY_FIRST_N = np.inf # short circuit cost matrix size to the first N samples, effectively limiting output length
#OBSCURE TUNABLES
SCORE_CLAMP_STEP = 10 # Controls how fast we increase the "gain" on cost matrix
CACHE_TRANSPOSE_MATRIX = True # Should just be default
#DEPRECATED
USE_BOX_CORNERS = False # Use older box corners point cloud approach, not as fine tunes
USE_SMOOTHED_DTW_PATH = False # Smooth the dtw path (via a moving average). Not ideal because it can move the path off the key points