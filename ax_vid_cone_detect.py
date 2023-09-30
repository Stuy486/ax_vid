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

def find_boxes_for_frame_torch(frame_np, v_ofs): # ([[boxes_np], [scores]], <model results>)
    if not hasattr(find_boxes_for_frame_torch, 'model'):
        find_boxes_for_frame_torch.model = YOLO(os.path.join(axv_files.paths['MODEL'], 'weights/best.pt'))
    
    ret = [[],[]]
    small = cv2.resize(frame_np, (1024, 576))
    left = small[160+v_ofs:416+v_ofs,:512,:]
    right = small[160+v_ofs:416+v_ofs,-512:,:]
    stacked = np.vstack((left, right))
    results = find_boxes_for_frame_torch.model.predict(stacked, verbose=False)
    boxes = results[0].boxes.cpu().numpy()
    for box in boxes:
        ret[0].append(box.xywh[0].tolist())
        ret[1].append(box.conf[0])
    return (ret, results)
    
def convert_xywh_to_x1y1x2y2(xywh):
    return [xywh[0] - (xywh[2] / 2), xywh[1] - (xywh[3] / 2), xywh[0] + (xywh[2] / 2), xywh[1] +(xywh[3] / 2)]

def process_video(axvid, data_hz, v_ofs): # returns (vid_id, vid_player, aud_player, [frames data, index is frame number])
    vid_player = axv_vid.VidPlayer(axvid)
    process_every_n_frames = axvid.fps / float(data_hz)

    ret_frame_data = [None] * axvid.num_frames
    
    vid_writer = None
    if OUTPUT_ANNOTATED_VIDEOS:
        annotated_vid_filename = os.path.join(axv_files.paths['VID_STORAGE'], axvid.yt_id + '_annotated.mp4')
        vid_writer = cv2.VideoWriter(annotated_vid_filename,
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     data_hz,
                                     (1024,256))

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
                (axvid.frame_data[this_frame], results) = find_boxes_for_frame_torch(frame, v_ofs)
            
            # Only return data at the requested data rate, even though we might have more cached
            ret_frame_data[this_frame] = axvid.frame_data[this_frame]
                
            if results != None and OUTPUT_ANNOTATED_VIDEOS:
                frame = results[0].plot()
                outframe = np.hstack([frame[:256,:,:], frame[256:,:,:]])
                vid_writer.write(outframe)
        else:
            vid_player.read()

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

    # unstack the cone positions (move cones in the bottom half to top, and right by the detection width)
    for frame_num in range(axvid.num_frames):
        data = ret_frame_data[frame_num]
        if data is not None:
            for i in range(len(data[0])):
                if data[0][i][0] >= 256:
                    data[0][i][0] -= 256
                    data[0][i][1] += 512

    if OUTPUT_ANNOTATED_VIDEOS:
        vid_writer.release()
    
    axv_vid.SaveVideo(axvid)

    return ret_frame_data

def save_cone_histograms(vid1_data, vid2_data, axvids):
    plots = []
    bucket_size = 4 # must be a power of 2
    max_y = []
    vertical_size = int(256/bucket_size)
    for vid_data in [vid1_data, vid2_data]:
        hist = [0] * vertical_size
        for data in vid_data:
            if data == None:
                continue
            for box in data[0]:
                box_y = min(vertical_size * 2, max(0, int(box[1]/bucket_size)))
                hist[box_y - vertical_size if box_y >= vertical_size else box_y] += 1
        
        hist = list(np.float_(hist) * (vertical_size / max(hist)))

        plot = np.zeros((vertical_size,int(max(hist)),3), dtype=np.uint8)
        for i, val in enumerate(hist):
            val = int(val)
            if val != 0:
                plot[i,:val-1,:] = 255
        plots.append(plot)
        max_y.append(hist.index(max(hist)))
    
    cv2.imwrite(axv_files.GetPairId(axvids) + "_vert.png", np.hstack(plots))

    y_adjustment = (max_y[0] - max_y[1]) * bucket_size
    for data in vid2_data:
        if data == None:
            continue
        for box in data[0]:
            box[1] += y_adjustment
            
    plots = []
    bucket_size = 8 # must be a power of 2
    max_x = []
    horizontal_size = int(1024/bucket_size)
    for vid_data in [vid1_data, vid2_data]:
        hist = [0] * horizontal_size
        for data in vid_data:
            if data == None:
                continue
            for box in data[0]:
                box_y = int(box[1])
                box_x = min(horizontal_size / 2, max(0, int(box[0]/bucket_size)))
                hist[box_x + int(horizontal_size / 2) if box_y >= 256 else box_x] += 1
        
        hist = list(np.float_(hist) * (horizontal_size / max(hist)))

        plot = np.zeros((int(max(hist)),horizontal_size,3), dtype=np.uint8)
        for i, val in enumerate(hist):
            val = int(val)
            if val != 0:
                plot[-val:,i,:] = 255
        plots.append(plot)
        max_x.append(hist.index(max(hist)))
    
    cv2.imwrite(axv_files.GetPairId(axvids) + "_horz.png", np.vstack(plots))
