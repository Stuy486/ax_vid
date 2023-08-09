#!/usr/bin/python3

import wx
import wx.lib.buttons as wxButtons
from pubsub import pub
import pytube
import cv2
import math
from threading import Thread
from ax_vid_cone_detect import process_video
from ax_vid_draw_frame import GenerateWithProcessedVids
import ax_vid_video as axv_vid
import ax_vid_files as axv_files
import ax_vid_progress as axv_prog
import ax_vid_cost_matrix as axv_cm

# Example URLs:

# nats '22 east
# https://www.youtube.com/watch?v=lY7soH93JWo Jeff  run 2
# https://www.youtube.com/watch?v=_zDkh9Jyz4E Jeff run 3
# https://www.youtube.com/watch?v=iwwVgGbugDQ Mack

# nats '21 west
# https://www.youtube.com/watch?v=nATH7DFgUaM Jeff
# https://www.youtube.com/watch?v=vB3ehSeEIrY McCelvey

# '23 Crow's NT day 1
# https://youtu.be/QT5RxHusUUg  Jeff  
# https://youtu.be/KMyUn61NxKU  Mack
# day 2
# https://youtu.be/yPEbldlLpbs Jeff
# https://youtu.be/tMnf6oxHLWQ Yon

# AAA 080722
# https://www.youtube.com/watch?v=ZZazCAWcZy4 Jeff 
# https://www.youtube.com/watch?v=wAjy2iLrYfg Tom L

# Lake Elsinore / Storm Stadium 062523
# https://www.youtube.com/watch?v=b4S-Q1oI_Nc Jeff
# https://www.youtube.com/watch?v=TJ7ckc-qrH0 Yon

# Setting these will auto-fill the text boxes for faster testing of changes
test_url1 = None
test_url2 = None
#test_url1 = 'https://www.youtube.com/watch?v=b4S-Q1oI_Nc'
#test_url2 = 'https://www.youtube.com/watch?v=TJ7ckc-qrH0'

def get_video_details(vid_player):
    fps = vid_player.get(cv2.CAP_PROP_FPS)
    num_frames = math.ceil(vid_player.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid_player.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_player.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, num_frames, (width, height)

ID_URL=0x10
ID_START=0x20
ID_FINISH=0x40

progress = False
def VidDownloadProgress(stream, chunk, bytes_remaining):
    global progress
    if not progress:
        wx.CallAfter(pub.sendMessage, axv_prog.START_PROGRESS,
                     title="Downloading...",
                     message="Downloading %s" % (stream.title),
                     max=stream.filesize)
        progress = True
    wx.CallAfter(pub.sendMessage, axv_prog.UPDATE_PROGRESS,
                 progress=stream.filesize-bytes_remaining)

def VidDownloadComplete(stream, file_path):
    global progress
    wx.CallAfter(pub.sendMessage, axv_prog.END_PROGRESS)
    progress = False

class AxButton(wxButtons.GenButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.SetForegroundColour(wx.WHITE)
        #self.SetBackgroundColour(wx.BLACK)

class VidElements(object):
    def __init__(self, parent, index):
        self.index = index
        self.instructions_label = wx.StaticText(parent, label="Paste the youtube URL of the %s video" %
                                                            ("first" if index == 0 else "second"))
        self.url_label = wx.StaticText(parent, label="Video %d URL:" % (index))
        self.url_box = wx.TextCtrl(parent, id=(ID_URL | index), size=(350, 25))
        self.start_label = wx.StaticText(parent, label="Start: Not set")
        self.start_button = AxButton(parent, id=(ID_START | index), label="Set Start")
        self.finish_label = wx.StaticText(parent, label="Finish: Not set")
        self.finish_button = AxButton(parent, id=(ID_FINISH | index), label="Set Finish")
    
    def ResetStartLabel(self):
        self.start_label.SetLabel("Start: Not set")
    
    def ResetFinishLabel(self):
        self.finish_label.SetLabel("Finish: Not set")

class VidComparison(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, title="Video Selection")

        self.ve = [VidElements(self, 0), VidElements(self, 1)]
        self.axvid = [None, None]
        self.vid1StartFrame = None
        self.vid1FinishFrame = None
        self.vid2StartFrame = None
        self.vid2FinishFrame = None

        self.topsizer = wx.BoxSizer(wx.VERTICAL)
        self.AddVideoElements(self.ve[0])
        self.AddVideoElements(self.ve[1])

        self.doneButton = AxButton(self, label="Generate Comparison Video")
        self.doneButton.Bind(wx.EVT_BUTTON, self.OnDoneButton)
        self.doneButton.Disable()
        self.topsizer.Add(self.doneButton, 1, wx.ALL, 5)

        pub.subscribe(self.StartProgress, axv_prog.START_PROGRESS)
        pub.subscribe(self.UpdateProgress, axv_prog.UPDATE_PROGRESS)
        pub.subscribe(self.EndProgress, axv_prog.END_PROGRESS)

        if test_url1 != None and test_url2 != None:
            self.ve[0].url_box.SetValue(test_url1)
            self.ve[1].url_box.SetValue(test_url2)

        self.SetSizerAndFit(self.topsizer)
        self.Show(True)
    
    def AddVideoElements(self, ve):
        self.topsizer.Add(ve.instructions_label, 0, wx.EXPAND | wx.TOP | wx.LEFT | wx.RIGHT, 5)

        horizontal = wx.BoxSizer(wx.HORIZONTAL)
        horizontal.Add(ve.url_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 5)
        ve.url_box.Bind(wx.EVT_TEXT, self.OnUrlChanged)
        horizontal.Add(ve.url_box, 1, wx.RIGHT, 5)
        self.topsizer.Add(horizontal, 0, wx.EXPAND | wx.TOP, 5)

        horizontal = wx.BoxSizer(wx.HORIZONTAL)
        buttonHorizontal = wx.BoxSizer(wx.HORIZONTAL)
        ve.start_button.Bind(wx.EVT_BUTTON, self.OnSetFrame)
        buttonHorizontal.Add(ve.start_label, 1, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 5)
        buttonHorizontal.Add(ve.start_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        horizontal.Add(buttonHorizontal, 1, 0)
        buttonHorizontal = wx.BoxSizer(wx.HORIZONTAL)
        ve.finish_button.Bind(wx.EVT_BUTTON, self.OnSetFrame)
        buttonHorizontal.Add(ve.finish_label, 1,wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        buttonHorizontal.Add(ve.finish_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        horizontal.Add(buttonHorizontal, 1, 0)
        self.topsizer.Add(horizontal, 0, wx.EXPAND | wx.TOP, 5)

    def StartProgress(self, title, message, max):
        axv_prog.StartGenericProgress(title, message, max)

    def UpdateProgress(self, progress):
        axv_prog.UpdateGenericProgress(progress)

    def EndProgress(self):
        axv_prog.EndGenericProgress()

    def OnUrlChanged(self, e):
        urlText = e.GetEventObject()
        urlId = urlText.GetId()
        index = urlId & 0xf
        ve = self.ve[index]
        try:
            vid = pytube.YouTube(ve.url_box.GetValue())
        except:
            msg = "Invalid video URL"
            self.axvid[index] = None
            ve.instructions_label.SetLabel(msg)
        else:
            vid.register_on_progress_callback(VidDownloadProgress)
            vid.register_on_complete_callback(VidDownloadComplete)
            axvid = axv_vid.LoadVideo(vid)
            self.axvid[index] = axvid
            Thread(target=axvid.Verify, args=(vid,)).start()
            #axvid.Verify(vid)
            ve.instructions_label.SetLabel("Set start/finish times")
            if hasattr(axvid, 'start_frame') and axvid.start_frame != None:
                self.SetStartFrame(ve, axvid.start_frame, save_changes=False)
            else:
                ve.ResetStartLabel()
            if hasattr(axvid, 'finish_frame') and axvid.finish_frame != None:
                self.SetFinishFrame(ve, axvid.finish_frame, save_changes=False)
            else:
                ve.ResetFinishLabel()
            self.MaybeEnableDoneButton()
    
    def OnDoneButton(self, e):
        #draw_frame_workers = axv_draw.start_drawing_workers()
        #vid_writer_worker = axv_draw.start_vid_out_worker(draw_frame_workers[4],
        #                                                  axv_files.GetPairOutputFile(self.axvid),
        #                                                  self.axvid[0].fps)
        Thread(target=DoVidCompare, args=(self.axvid,)).start()
    
    def MaybeEnableDoneButton(self):
        enable = True
        for axvid in self.axvid:
            if axvid is None or not axvid.HasFramesSet():
                enable = False
        self.doneButton.Enable(enable)
    
    def SetStartFrame(self, ve, start_frame, save_changes=True):
        axvid = self.axvid[ve.index]
        axvid.start_frame = start_frame
        ve.start_label.SetLabel("Start: %d" % (start_frame))
        if axvid.finish_frame != None:
            ve.instructions_label.SetLabel("%s video is ready" % ("First" if ve.index == 0 else "Second"))
            if save_changes:
                axv_vid.SaveVideo(axvid)

    def SetFinishFrame(self, ve, finish_frame, save_changes=True):
        axvid = self.axvid[ve.index]
        axvid.finish_frame = finish_frame
        ve.finish_label.SetLabel("Finish: %d" % (finish_frame))
        if axvid.start_frame != None:
            ve.instructions_label.SetLabel("%s video is ready" % ("First" if ve.index == 0 else "Second"))
            if save_changes:
                axv_vid.SaveVideo(axvid)
    
    def OnSetFrame(self, e):
        buttonId = e.GetEventObject().GetId()
        index = buttonId & 0xf
        id = buttonId & 0xf0
        start = (id == ID_START)
        ve = self.ve[index]
        axvid = self.axvid[index]
        if axvid is None:
            return
        with VidFrameSelector(self, axvid, start) as dlg:
            if dlg.ShowModal() == True:
                if start:
                    self.SetStartFrame(ve, dlg.cur_frame)
                else:
                    self.SetFinishFrame(ve, dlg.cur_frame)
                self.MaybeEnableDoneButton()


class VidFrameSelector(wx.Dialog):
    def __init__(self, parent, axvid, start):
        wx.Dialog.__init__(self, parent, title="Video Frame Selector")

        self.topsizer = wx.BoxSizer(wx.VERTICAL)

        self.axvid = axvid
        self.vid_player = axv_vid.VidPlayer(axvid)

        self.instructions_label = wx.StaticText(self,
            label="Seek to the frame in the video where the car crosses the %s line." %
                                                            ("start" if start else "finish"))
        self.topsizer.Add(self.instructions_label, 50, wx.ALIGN_CENTER | wx.ALL, 5)

        self.vid_view = wx.GenericStaticBitmap(self)
        self.vid_view.SetScaleMode(wx.GenericStaticBitmap.Scale_AspectFit)
        self.topsizer.Add(self.vid_view, 720, wx.EXPAND | wx.ALL, 5)

        self.slider = wx.Slider(self, minValue=0, maxValue=500)
        self.slider.Bind(wx.EVT_SLIDER, self.OnSliderScroll)
        self.topsizer.Add(self.slider, 50, wx.EXPAND | wx.ALL, 5)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.back_button = AxButton(self, label="Back 1 Frame")
        self.back_button.Bind(wx.EVT_BUTTON, self.OnBackButton)
        button_sizer.Add(self.back_button, 2, wx.EXPAND | wx.ALL, 5)

        self.forward_button = AxButton(self, label="Forward 1 Frame")
        self.forward_button.Bind(wx.EVT_BUTTON, self.OnForwardButton)
        button_sizer.Add(self.forward_button, 2, wx.EXPAND | wx.ALL, 5)

        self.done_button = AxButton(self, label="Done")
        self.done_button.Bind(wx.EVT_BUTTON, self.OnDoneButton)
        button_sizer.Add(self.done_button, 1, wx.EXPAND | wx.ALL, 5)

        self.topsizer.Add(button_sizer, 50, wx.EXPAND, 0)

        if start and axvid.start_frame is not None:
            self.SetVideoFrame(axvid.start_frame)
            self.slider.SetValue(int(axvid.start_frame / axvid.num_frames * 500))
        elif not start and axvid.finish_frame is not None:
            self.SetVideoFrame(axvid.finish_frame)
            self.slider.SetValue(int(axvid.finish_frame / axvid.num_frames * 500))
        else:
            self.SetVideoFrame(0)

        self.SetSizerAndFit(self.topsizer)
        self.Show(True)

    def OnBackButton(self, e):
        self.SetVideoFrame(self.cur_frame - 1)

    def OnForwardButton(self, e):
        self.SetVideoFrame(self.cur_frame + 1)

    def OnSliderScroll(self, e):
        self.SetVideoFrame(self.slider.GetValue() * self.axvid.num_frames / 500)

    def OnDoneButton(self, e):
        self.EndModal(True)
    
    def SetVideoFrame(self, frame_index):
        self.cur_frame = int(frame_index)
        self.vid_player.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        _,frame = self.vid_player.read()

        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.vid_view.SetBitmap(wx.Bitmap.FromBuffer(1280, 720, frame))
        self.topsizer.Layout()
    
    def OnExit(self,e):
        self.Close(True)  # Close the frame.

def DoVidCompare(axvids):
    vid1_data = process_video(axvids[0])
    vid2_data = process_video(axvids[1])
    cm = axv_cm.GetCostMatrix(vid1_data, vid2_data, axv_files.GetPairId(axvids))
    GenerateWithProcessedVids(axvids, cm)

if __name__ == '__main__':
    axv_files.CreateDirectories()
    app = wx.App(False) # Create a new app, don't redirect stdout/stderr to a window.
    frame = VidComparison(None)
    app.MainLoop()