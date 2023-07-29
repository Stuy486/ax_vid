from cx_Freeze import setup, Executable
import sys

# Dependencies are automatically detected, but it might need
# fine tuning.
build_options = {
    'packages': [],
    'excludes': [],
    'include_files': [('fonts', 'fonts'),
                      ('yolov8_cone_detection_model', 'yolov8_cone_detection_model'),
                      ('default.yaml', 'ultralytics/cfg/')]
}

base = 'console'

executables = [
    Executable('ax_vid_gui.py', base=base, target_name = 'ax_vid')
]

sys.setrecursionlimit(1000)

setup(name='ax_vid',
      version = '1.0',
      description = '',
      options = {'build_exe': build_options},
      executables = executables)
