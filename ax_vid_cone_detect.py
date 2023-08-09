import os
from ultralytics import YOLO
import cv2
import numpy as np
#import cProfile
#import pstats
from ax_vid_config import *
import ax_vid_video as axv_vid
import ax_vid_files as axv_files
import ax_vid_progress as axv_prog

def find_boxes_for_frame_torch(frame_np): # ([[boxes_np], [scores]], <model results>)
    if not hasattr(find_boxes_for_frame_torch, 'model'):
        find_boxes_for_frame_torch.model = YOLO(os.path.join(axv_files.paths['MODEL'], 'weights/best.pt'))
    
    ret = [[],[]]
    small = cv2.resize(frame_np, (1024, 576))
    left = small[160:416,:512,:]
    right = small[160:416,-512:,:]
    stacked = np.vstack((left, right))
    results = find_boxes_for_frame_torch.model.predict(stacked, verbose=False)
    boxes = results[0].boxes.cpu().numpy()
    for box in boxes:
        ret[0].append(box.xywh[0].tolist())
        ret[1].append(box.conf[0])
    return (ret, results)
    
def convert_xywh_to_x1y1x2y2(xywh):
    return [xywh[0] - (xywh[2] / 2), xywh[1] - (xywh[3] / 2), xywh[0] + (xywh[2] / 2), xywh[1] +(xywh[3] / 2)]

def process_video(axvid): # returns (vid_id, vid_player, aud_player, [frames data, index is frame number])
    vid_player = axv_vid.VidPlayer(axvid)
    process_every_n_frames = axvid.fps / float(DATA_HZ)

    ret_frame_data = [None] * axvid.num_frames
    
    vid_writer = None
    if OUTPUT_ANNOTATED_VIDEOS:
        annotated_vid_filename = os.path.join(axv_files.paths['VID_STORAGE'], axvid.yt_id + '_annotated.mp4')
        vid_writer = cv2.VideoWriter(annotated_vid_filename,
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     DATA_HZ,
                                     (512,512))

    frame_num = 0
    next_processed_frame = axvid.start_frame
    vid_player.set(cv2.CAP_PROP_POS_FRAMES, axvid.start_frame)

    axv_prog.StartProgress('Finding cones',
                                  'Finding cones (%.2f fps, %d frames, %dx%d)' %
                                  (axvid.fps, axvid.num_frames, axvid.width, axvid.height),
                                  axvid.num_frames - axvid.start_frame)
    while True:
        this_frame = axvid.start_frame + frame_num
        axv_prog.UpdateProgress(frame_num)
        if this_frame > axvid.finish_frame:
            break
        if this_frame >= next_processed_frame or this_frame == axvid.finish_frame:
            next_processed_frame += process_every_n_frames
            results = None
            frame = None
            if axvid.frame_data[this_frame] == None or OUTPUT_ANNOTATED_VIDEOS:
                success,frame = vid_player.read()
                if not success:
                    print("Error reading frame: %d" % (this_frame))
            if OUTPUT_ANNOTATED_VIDEOS or (axvid.frame_data[this_frame] is None and frame is not None):
                (axvid.frame_data[this_frame], results) = find_boxes_for_frame_torch(frame)
            
            # Only return data at the requested data rate, even though we might have more cached
            ret_frame_data[this_frame] = axvid.frame_data[this_frame]
                
            if results != None and OUTPUT_ANNOTATED_VIDEOS:
                vid_writer.write(results[0].plot())

        frame_num += 1
    axv_prog.EndProgress()

    average_cones_per_frame = None
    conf_threshold = .30
    while average_cones_per_frame is None or average_cones_per_frame > TARGET_CONES_PER_FRAME:
        cone_count = 0
        frame_count = 0
        for frame_num in range(axvid.num_frames):
            data = ret_frame_data[frame_num]
            if data is not None:
                new_data = [[],[]]
                frame_count += 1
                for i in range(len(data[0])):
                    if data[1][i] > conf_threshold:
                        cone_count += 1
                        new_data[0].append(data[0][i])
                        new_data[1].append(data[1][i])
                ret_frame_data[frame_num] = new_data
        average_cones_per_frame = cone_count / frame_count
        print('%.2f cones per frame at %.2f confidence' % (average_cones_per_frame, conf_threshold))
        conf_threshold += .01

    if OUTPUT_ANNOTATED_VIDEOS:
        vid_writer.release()
    
    axv_vid.SaveVideo(axvid)

    return ret_frame_data