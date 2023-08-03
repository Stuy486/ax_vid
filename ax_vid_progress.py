import wx
import time
from pubsub import pub

START_PROGRESS = 'progress.start'
UPDATE_PROGRESS = 'progress.update'
END_PROGRESS = 'progress.end'

def StartProgress(title, message, max):
    wx.CallAfter(pub.sendMessage, START_PROGRESS,
                 title=title, message=message, max=max)
last_call = 0
def UpdateProgress(progress):
    global last_call
    now = time.time()
    if now - last_call > .5:
        wx.CallAfter(pub.sendMessage, UPDATE_PROGRESS,
                     progress=progress)
        last_call = now
def EndProgress():
    wx.CallAfter(pub.sendMessage, END_PROGRESS)

generic_progress = None
generic_progress_max = 100
def StartGenericProgress(title, message, max):
    global generic_progress, generic_progress_max
    if generic_progress is not None:
        generic_progress.Close()
    generic_progress = wx.ProgressDialog(title, message)
    generic_progress_max = max

def UpdateGenericProgress(progress):
    global generic_progress, generic_progress_max
    if generic_progress is not None:
        generic_progress.Update(int(100 * progress / generic_progress_max))

def EndGenericProgress():
    global generic_progress
    if generic_progress is not None:
        generic_progress.Close()
        generic_progress = None