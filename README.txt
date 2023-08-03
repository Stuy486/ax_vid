#### INSTALL DEPENDENCIES ####

The main requirement is that you install python, which you can get here:
https://www.python.org/downloads/

For me on OSX, after installation, the installer opened a directory that had a
couple of scripts. I needed to run two of these for this program to work:
"Install Certificates"
"Update Shell Profile"

After python is installed (and your shell profile is updated), you should be
able to run pip, which will install the rest of the dependencies (in a command
prompt in Windows, or terminal in OSX):
> python3 -m pip install dtw-python ultralytics numpy Pillow progressbar scikit-learn wxPython pytube moviepy pypubsub dill

If it complains that python3 isn't a command, the "Update Shell Profile" script
didn't work properly, and you need to add python to your PATH.

For sounds to be added to the video, you'll need ffmpeg installed:
https://ffmpeg.org/download.html

The program will work without ffmpeg, but it will not generate videos with
sound.

#### RUNNING THE PROGRAM ####

Once python is in your path, and everything is pip installed, you should be
able to run the program in one of two ways:

-- From the terminal/command prompt --
> cd <path/to/ax_vid>
> python3 ax_vid_gui.py

-- Using "Python Launcher" --
The python installer should've installed a program called "Python Launcher". If
you run Python Launcher, you can then run ax_vid_gui.py from that by selecting
File->Open->ax_vid_gui.py

#### USING THE PROGRAM ####

You should see the main window, which has two text boxes. Paste the URL of each
YouTube video in these. It will download the video when you paste.

Once the videos are downloaded, you need to set the start and finish for each
run. Click the "Set Start" and "Set Finish" buttons to do this. It will open
another window showing the video. Use the slide bar to get it close, then the
backwards/forwards 1 frame buttons to get it exact. The idea is to get car
aligned in the same location in each video. If these are misaligned relative to
each other, the result video will be wrong. Click the Done button when you have
it set.

Once all for start/finishes are set, you can click the "Generate Comparison
Video" button. After that, it's mostly a bunch of waiting. Here's a brief
description of each step:.

Cone Detection:
First it look at the frames of each video and identify cones using object
detection. The placement of cones in the frame is how it aligns the two videos.

Cost Matrix Creation:
The "cost matrix" is a grid comparing each from of video 1 to each frame of
video 2. Because the comparison is somewhat slow, it only generates the costs
more or less along the diagonal of the cost matrix. You can see these visually
in the cost_matrices/ directory.

Time Slip Video Creation:
Using the cost matrix, the two videos are aligned with the goal of having the
car in the same place on the track in each video. The difference between the
timestamps of the aligned frames is the time slip, i.e. how much time video 1
gained/lost to video 2 up to that point on the track.

Joining of Video/Audio
The video creation step generates a silent video. This step takes the audio of
the left video and stitches it together with the silent video to create a video
with sound. Since the right video is being sped up/slowed down to match the
left, its audio would sound very weird (and also I don't know how to do it).

Once the process completes, the videos should appear in the output_videos/
directory.
