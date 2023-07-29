import os
import sys

paths = {
    'VID_STORAGE': 'vid_storage',
    'SAVED_VIDEO_DATA': 'saved_vid_data',
    'COST_MATRICES': 'cost_matrices',
    'OUTPUT_VIDEOS': 'output_videos',
    'FONTS': 'fonts',
    'MODEL': 'yolov8_cone_detection_model',
    'ULTRALYTICS_CFG': 'ultralytics_cfg.yaml'
}

osx_libs = [
    'libtorch.dylib',
    'libc10.dylib',
    'libtorch_python.dylib',
    'libtorch_cpu.dylib'
]

def CreateDirectories():
    base_dir = os.path.dirname(__file__)
    print(base_dir)

    for name, path in paths.items():
        abs_path = os.path.join(base_dir, path)
        if not os.path.exists(abs_path):
            os.mkdir(abs_path)
        paths[name] = abs_path

    for lib in osx_libs:
        abs_path = os.path.join(base_dir, lib)
        if os.path.exists(abs_path) and not os.path.islink(abs_path):
            os.remove(abs_path)
            os.symlink(os.path.join(base_dir, 'torch', 'lib', lib), abs_path)