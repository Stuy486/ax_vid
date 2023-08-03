import os
import pickle
import pytube
import cv2
import math
from ax_vid_files import paths

def MetadataFilepath(yt_id):
    return os.path.join(paths['SAVED_VIDEO_DATA'], yt_id + '.axvd')

def LoadVideo(pt_vid):
    metadata_filepath = MetadataFilepath(pt_vid.video_id)
    if os.path.exists(metadata_filepath):
        with open(metadata_filepath, 'rb') as metadata_file:
            return pickle.load(metadata_file)
    else:
        return Video(pt_vid)

def SaveVideo(axvid):
    with open(MetadataFilepath(axvid.yt_id), 'wb') as metadata_file:
        pickle.dump(axvid, metadata_file)

vid_players = {}
def VidPlayer(axvid, pt_vid=None):
    if axvid not in vid_players:
        if pt_vid is None:
            pt_vid = axvid.GetPyTube()
        vid_filename = os.path.join(paths['VID_STORAGE'], pt_vid.video_id + '.mp4')
        if not os.path.exists(vid_filename):
            pt_vid.streams.order_by("resolution").filter(adaptive=True,file_extension='mp4')[-1].download(filename=vid_filename)
        vid_players[axvid] = cv2.VideoCapture(vid_filename)
    return vid_players[axvid]

aud_filenames = {}
def AudFilename(axvid, pt_vid=None):
    if axvid not in aud_filenames:
        if pt_vid is None:
            pt_vid = axvid.GetPyTube()
        aud_filename = os.path.join(paths['VID_STORAGE'], pt_vid.video_id + '.m4a')
        if not os.path.exists(aud_filename):
            pt_vid.streams.filter(adaptive=True,only_audio=True)[0].download(filename=aud_filename)
        aud_filenames[axvid] = aud_filename
    return aud_filenames[axvid]

class Video(object):
    def __init__(self, pt_vid):
        self.yt_id = pt_vid.video_id

    def Verify(self, pt_vid):
        vid_player = VidPlayer(self, pt_vid)
        aud_filenames = AudFilename(self, pt_vid)
        self.fps = vid_player.get(cv2.CAP_PROP_FPS)
        self.num_frames = math.ceil(vid_player.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(vid_player.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(vid_player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.title = pt_vid.title

        if not hasattr(self, 'frame_data') or len(self.frame_data) != self.num_frames:
            self.frame_data = [None] * self.num_frames
            self.start_frame = None
            self.finish_frame = None
    
    def HasFramesSet(self):
        return hasattr(self, 'start_frame') and hasattr(self, 'finish_frame') and self.start_frame is not None and self.finish_frame is not None
    
    def StartInMS(self):
        if hasattr(self, 'start_frame'):
            return (1000 / self.fps * self.start_frame)
    
    def GetPyTube(self):
        return pytube.YouTube("https://youtu.be/" + self.yt_id)