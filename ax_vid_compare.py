
import math
import os
from ultralytics import YOLO
import cv2
import numpy as np
import numpy.polynomial.polynomial as poly
import icp
import dtw
import moviepy.editor as mpe
import cpuinfo
from threading import Thread, Event
from queue import Queue, Full, Empty
from PIL import Image, ImageDraw, ImageFont, ImageColor
from progressbar import Bar, ETA, ProgressBar, Percentage, RotatingMarker
from sklearn.neighbors import NearestNeighbors
#import cProfile
#import pstats
import ax_vid_video
from ax_vid_files import paths
import ax_vid_progress

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

def find_boxes_for_frame_torch(frame_np): # ([[boxes_np], [scores]], <model results>)
    if not hasattr(find_boxes_for_frame_torch, 'model'):
        find_boxes_for_frame_torch.model = YOLO(os.path.join(paths['MODEL'], 'weights/best.pt'))
    
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

def flatten(l):
    return [item for sublist in l for item in sublist]

def calculate_frame_similarity(data1, data2):
    if not hasattr(calculate_frame_similarity, 'cached_T'):
        calculate_frame_similarity.cached_T = None  # it doesn't exist yet, so initialize it
    num_A = len(data1[0])
    num_B = len(data2[0])
    A = np.asarray(data1[0])
    B = np.asarray(data2[0])
    # If one frame has more datapoints than the other, prune the furthest
    # (nearest neighbor) points from the longer list
    # print(numA, numB, A.shape, B.shape)
    if num_A != num_B:
        if num_A > num_B:
            A, B = B, A
            num_A, num_B = num_B, num_A
        if True:
            B = B[:len(A)]
        else:
            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(A)
            dist,_ = neigh.kneighbors(B)
            distances = sorted(zip(flatten(dist),range(num_B)), key=lambda x:x[0], reverse=True)

            remove = []
            for i in range(num_B - num_A):
                remove.append(distances[i][1])

            remove.sort(reverse=True)
            B = np.delete(B, remove, 0)

            # if len(distances) == 1:
            #     return distances[0]
            # zscores = abs(stats.zscore(distances))
            # total = 0
            # num = 0
            # for i, zscore in enumerate(zscores):
            #     if zscore < 1 or distances[i] < 30:
            #         total += distances[i]
            #         num += 1
            
            #     return 255 if num == 0 else total / num
    
    if USE_BOX_CORNERS:
        new_A = []
        new_B = []
        for box in A:
            new_A.append([box[0], box[1]])
            new_A.append([box[0] + box[2], box[1]])
            new_A.append([box[0], box[1] + box[3]])
            new_A.append([box[0] + box[2], box[1] + box[3]])
        for box in B:
            new_B.append([box[0], box[1]])
            new_B.append([box[0] + box[2], box[1]])
            new_B.append([box[0], box[1] + box[3]])
            new_B.append([box[0] + box[2], box[1] + box[3]])
        T, t, distances, indices, iters = icp.icp(np.asarray(new_A), np.asarray(new_B),
                                              init_pose=calculate_frame_similarity.cached_T)
        if CACHE_TRANSPOSE_MATRIX:
            calculate_frame_similarity.cached_T = T
        distances_weight = .4 / (6 - min(5, num_A))
        distances_weight += .6
        translation_distance = math.sqrt((t[0]*t[0])+(t[1]*t[1]))
        return (np.median(distances) * distances_weight) + (translation_distance * (1.0 - distances_weight))
    else:
        # Calculate weights for each cone (based on cone box area)
        total_A_weight = 0
        total_B_weight = 0
        for box in A:
            total_A_weight += box[2]*box[3]*box[2]*box[3]
        for box in B:
            total_B_weight += box[2]*box[3]*box[2]*box[3]
        T, t, distances, indices, iters = icp.icp(A[:,:2], B[:,:2],
                                              init_pose=calculate_frame_similarity.cached_T)
        if CACHE_TRANSPOSE_MATRIX:
            calculate_frame_similarity.cached_T = T
        distance_weights = [0] * len(distances)
        # Give the distances for larger cones more weight
        for i, a_box in enumerate(A):
            b_box = B[indices[i]]
            a_box_weight = (a_box[2] * a_box[3] * a_box[2] * a_box[3]) / total_A_weight
            b_box_weight = (b_box[2] * b_box[3] * b_box[2] * b_box[3]) / total_B_weight
            distance_weights[i] = a_box_weight + b_box_weight
        distances = [distances[i] * distance_weights[i] for i in range(len(distances))]
        # Give individual point distances more weight if there are more cones in the frame
        distances_weight = .1 * min(8, num_A - 1)
        translation_distance = math.sqrt((t[0]*t[0])+(t[1]*t[1]))
        return (np.mean(distances) * distances_weight) + (translation_distance * (1.0 - distances_weight))
    
def calculate_frame_similarity2(data1, data2):
    scores = []
    for a, a_conf in zip(data1[0], data1[1]):
        closest = None
        closest_conf = None
        closest_dist = None
        for b, b_conf in zip(data2[0], data2[1]):
            dist = math.dist(a[0:1],b[0:1])
            if closest_dist is None or dist < closest_dist:
                closest = b
                closest_conf = b_conf
                closest_dist = dist
        scores.append((closest_dist + 10 * (abs(a[2]-closest[2]) + abs(a[3]-closest[3]))) / (a_conf * closest_conf))

    scores.sort()
    scores = scores[:len(data1[0])]
    
    return sum(scores) / len(scores)
    
def convert_xywh_to_x1y1x2y2(xywh):
    return [xywh[0] - (xywh[2] / 2), xywh[1] - (xywh[3] / 2), xywh[0] + (xywh[2] / 2), xywh[1] +(xywh[3] / 2)]

def get_video_details(vid_player):
    fps = vid_player.get(cv2.CAP_PROP_FPS)
    num_frames = math.ceil(vid_player.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid_player.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_player.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, num_frames, (width, height)

def process_video(axvid): # returns (vid_id, vid_player, aud_player, [frames data, index is frame number])
    vid_player = ax_vid_video.VidPlayer(axvid)
    process_every_n_frames = axvid.fps / float(DATA_HZ)

    ret_frame_data = [None] * axvid.num_frames
    
    vid_writer = None
    if OUTPUT_ANNOTATED_VIDEOS:
        annotated_vid_filename = os.path.join(paths['VID_STORAGE'], axvid.yt_id + '_annotated.mp4')
        vid_writer = cv2.VideoWriter(annotated_vid_filename,
                                     cv2.VideoWriter_fourcc(*'mp4v'),
                                     DATA_HZ,
                                     (512,512))

    frame_num = 0
    next_processed_frame = axvid.start_frame
    vid_player.set(cv2.CAP_PROP_POS_FRAMES, axvid.start_frame)

    ax_vid_progress.StartProgress('Finding cones',
                                  'Finding cones (%.2f fps, %d frames, %dx%d)' %
                                  (axvid.fps, axvid.num_frames, axvid.width, axvid.height),
                                  axvid.num_frames - axvid.start_frame)
    bar = ProgressBar(widgets=['Finding cones (%.2f fps, %d frames, %dx%d)' %
                                (axvid.fps, axvid.num_frames, axvid.width, axvid.height),
                               Percentage(), ' ',
                               Bar(marker=RotatingMarker()), ' ',
                               ETA()], maxval=axvid.num_frames - axvid.start_frame).start()
    while True:
        this_frame = axvid.start_frame + frame_num
        ax_vid_progress.UpdateProgress(frame_num)
        bar.update(frame_num)
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
    ax_vid_progress.EndProgress()
    bar.finish()

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
    
    ax_vid_video.SaveVideo(axvid)

    return (axvid, vid_player, ret_frame_data)

def find_path_through_cost_matrix(cost_matrix):
    step = SCORE_CLAMP_STEP
    score_clamp = cost_matrix.min()
    average_along_dtw = 255
    # The provided cost matrix has not been "clamped", meaning that full range from minimum to maximum
    # frame comparison scores could be very large. In general, we only care about the scores that are
    # very low, meaning the cones lined up very well. This loop clamps all the values in the raw cost
    # matrix to be within a maximum value. Once the values are clamped, they are normalized to fill
    # the range 0-255. The DTW path is then found, and the average cost along the path is found. We
    # raise the max clamping value until the average along the path is below a certain threshold. This
    # serves to increase the gain to similar levels between well aligned and poorly aligned videos.
    while average_along_dtw > TARGET_DTW_AVERAGE_SCORE:
        clamped_cm = np.where(cost_matrix > score_clamp, score_clamp, cost_matrix)
        clamped_cm *= 255.0/clamped_cm.max()
        dtw_path = dtw.dtw(clamped_cm, step_pattern='typeIbs')
        prev_average = average_along_dtw
        average_along_dtw = np.mean([clamped_cm[i1][i2] for (i1,i2) in zip(dtw_path.index1, dtw_path.index2)])
        if average_along_dtw - prev_average > 2:
            step = max(step / 2, SCORE_CLAMP_STEP)
        else:
            step *= 2
        score_clamp += step
        if score_clamp >= cost_matrix.max():
            break
    return dtw_path

def get_initial_cost_matrix(vid1_id, vid1_data, vid2_id, vid2_data):
    # This should probably change, but for now, we're just creating the cost matrix so it has one
    # entry on each axis for every frame that we have cone data for. This means that for the cost
    # matrix to be monotonic along each axis, the frame data provided must be evenly spaced. This
    # is why the frame data returned by process_video only includes datapoints at the requested
    # DATA_HZ
    data1_matrix_dim = 0
    for data1 in vid1_data:
        if data1 != None:
            data1_matrix_dim += 1
            
    data2_matrix_dim = 0
    for data2 in vid2_data:
        if data2 != None:
            data2_matrix_dim += 1

    data1_matrix_dim = min(data1_matrix_dim, ONLY_FIRST_N)
    data2_matrix_dim = min(data2_matrix_dim, ONLY_FIRST_N)
    cost_matrix = np.ones([data1_matrix_dim,data2_matrix_dim]) * np.inf

    ax_vid_progress.StartProgress('Cost matrix',
                                  'Calculating cost matrix (%dx%d) ' % (data1_matrix_dim, data2_matrix_dim),
                                  data1_matrix_dim)
    bar = ProgressBar(widgets=['Calculating cost matrix (%dx%d) ' % (data1_matrix_dim, data2_matrix_dim),
                               Percentage(), ' ',
                               Bar(marker=RotatingMarker()), ' ',
                               ETA()], maxval=data1_matrix_dim).start()

    current_time_slip_estimate = 0 # in DATA_HZ increments, i.e. vid_data[n]
    time_slip_check_interval = COST_MATRIX_RECENTER_INTERVAL * DATA_HZ # In DATA_HZ increments
    next_time_slip_check = time_slip_check_interval

    max_score = 0
    data1_matrix_dim = 0
    max_comparison_window = time_slip_check_interval / (1 / MAX_TIME_SLIP_PER_SECOND)
    comparison_window_increment = max_comparison_window / time_slip_check_interval
    current_comparison_window = DATA_HZ / 4.0
    #prof = cProfile.Profile()
    #prof.enable()
    for data1 in vid1_data:
        if data1 == None:
            continue
        ax_vid_progress.UpdateProgress(data1_matrix_dim + 1)
        bar.update(data1_matrix_dim + 1)
        # As we calculate the cost matrix, calculate intermediate estimated time slip
        # Use this to offset which frames we're looking at in vid2. This allows us to
        # shift the comparison window arbitrarily (not just a window around the
        # diagonal), and also lets us use a much smaller comparison window.
        data2_offset_dim = data1_matrix_dim + current_time_slip_estimate
        if data1_matrix_dim > next_time_slip_check:
            next_time_slip_check += time_slip_check_interval
            if max_score != 0:
                #partial_cm = cost_matrix[:data1_matrix_dim, :data2_offset_dim]
                #partial_cm = np.where(partial_cm == np.inf, max_score, partial_cm)
                partial_cm = np.where(cost_matrix == np.inf, max_score, cost_matrix)
                current_dtw_path = find_path_through_cost_matrix(partial_cm)
                # look about 3/4ths of the way through the check interval, as the path tends
                # to curve back to the corner at the end.
                #rewind = int(time_slip_check_interval * 3 / 4)
                #current_time_slip_estimate = current_dtw_path.index2[-rewind] - current_dtw_path.index1[-rewind]
                index = data1_matrix_dim - int(time_slip_check_interval * 1 / 4)
                current_time_slip_estimate = current_dtw_path.index2[index] - current_dtw_path.index1[index]
                data2_offset_dim = data1_matrix_dim + current_time_slip_estimate
                current_comparison_window = DATA_HZ / 4.0
        current_comparison_window += comparison_window_increment
        if data2_offset_dim > data2_matrix_dim:
            # If the diagonal projection of the current time slip estimate runs off the
            # end of data2 (to the right), then stop traversing further rows in data1.
            # Assuming our video correlation is correct-ish, this limits parsing the
            # end of data1 in the case where video1 is significantly longer than video2
            # (after the finish line)
            break
        if len(data1[0]) > 0:
            data2_matrix_dim = 0
            for data2 in vid2_data:
                if data2 == None:
                    continue
                if abs(data2_offset_dim - data2_matrix_dim) < current_comparison_window and data1_matrix_dim < ONLY_FIRST_N and data2_matrix_dim < ONLY_FIRST_N:
                    if len(data2[0]) > 0:
                        #score = calculate_frame_similarity(data1,data2)
                        score = calculate_frame_similarity2(data1,data2)
                        max_score = max(score, max_score)
                        cost_matrix[data1_matrix_dim][data2_matrix_dim] = score
                data2_matrix_dim += 1
        data1_matrix_dim += 1
    #prof.disable()
    #stats = pstats.Stats(prof).strip_dirs().sort_stats("tottime")
    #stats.print_stats(50)
    
    ax_vid_progress.EndProgress()
    bar.finish()

    #cost_matrix = cost_matrix[:data1_matrix_dim,:data2_offset_dim]
    cost_matrix = np.where(cost_matrix == np.inf, max_score, cost_matrix)

    score_clamp = cost_matrix.min() + 10
    average_along_dtw = 255
    while average_along_dtw > 200:
        clamped_cm = np.where(cost_matrix > score_clamp, score_clamp, cost_matrix)
        clamped_cm *= 255.0/clamped_cm.max()
        test_dtw = dtw.dtw(clamped_cm, step_pattern='typeIbs')
        average_along_dtw = np.mean([clamped_cm[i1][i2] for (i1,i2) in zip(test_dtw.index1, test_dtw.index2)])
        score_clamp += SCORE_CLAMP_STEP
        if score_clamp >= cost_matrix.max():
            break

    return clamped_cm

def identify_key_frames(initial_cost_matrix):
    alignment = dtw.dtw(initial_cost_matrix, step_pattern='typeIbs')

    path = list(zip(alignment.index1, alignment.index2))
    if USE_SMOOTHED_DTW_PATH:
        smoothed_path = []
        smothing_window = DATA_HZ
        for i in range(len(path)):
            start = max(0, i - int(smothing_window / 2))
            end = min(len(path) - 1, start + smothing_window)
            i1_avg = int(np.mean([i1 for (i1,_) in path[start:end]]))
            i2_avg = int(np.mean([i2 for (_,i2) in path[start:end]]))
            smoothed_path.append((i1_avg,i2_avg))
        path = smoothed_path
    minimum_key_delta = int(DATA_HZ * MIN_KEY_FRAME_DELTA)
    maximum_key_delta = int(DATA_HZ * MAX_KEY_FRAME_DELTA)
    key_frames = []

    last = 0
    best_since_last = 0
    best_since_last_val = np.inf
    for i, (i1,i2) in enumerate(path):
        new_key_frame = None
        val = initial_cost_matrix[i1][i2]
        if i - last >= minimum_key_delta and val < best_since_last_val:
            best_since_last = i
            best_since_last_val = val
        if val < (255 - KEY_FRAME_COST_THRESHOLD) and i - last >= minimum_key_delta:
            new_key_frame = i
        elif best_since_last != 0 and i - last > maximum_key_delta:
            new_key_frame = best_since_last
        if new_key_frame:
            key_frames.append(path[new_key_frame])
            last = new_key_frame
            best_since_last = 0
            best_since_last_val = np.inf

    #if best_since_last != 0 and len(path) - last > minimum_key_delta:
    #    key_frames.append(path[best_since_last])
    if (abs(path[-1][0] - key_frames[-1][0]) / DATA_HZ) > (MIN_KEY_FRAME_DELTA * 2):
        key_frames.append(path[-1])
    else:
        key_frames[-1] = path[-1]

    return key_frames

def get_key_frame_poly_data(key_frames): # [(vid1_ts_s, vid2_ts_s)], [coefs]
    kf_ts = np.asarray([(0,0)] + [
        (1000.0 * vid1_kf_index * (1 / DATA_HZ),
         1000.0 * vid2_kf_index * (1 / DATA_HZ))
         for (vid1_kf_index, vid2_kf_index) in key_frames])
    kf_coefs = [None] * len(kf_ts)
    for i in range(len(kf_ts)):
        slice = kf_ts[max(0,i-1):min(len(kf_ts),i+2)]
        x = slice[:,0]
        y = slice[:,1]
        kf_coefs[i] = poly.polyfit(x, y, len(slice) - 1)
    return kf_ts, kf_coefs

TIMING_Y = 0 # Offset from base of middle image
SLIP_BAR_Y = TIMING_Y + TIMING_HEIGHT
SLIP_TEXT_Y = SLIP_BAR_Y + SLIP_BAR_HEIGHT
SLIP_GRAPH_Y = SLIP_TEXT_Y + SLIP_TEXT_HEIGHT

def get_time_slip_graph(axvid1, key_frames):
    num_vid1_frames = key_frames[-1][0]
    time_slip_data = []
    if USE_POLY_KEY_FRAME_SMOOTHING:
        kf_ts, kf_poly_coefs = get_key_frame_poly_data(key_frames)
        for vid1_frame in range(num_vid1_frames):
            vid1_ofs_ms = vid1_frame * (1000.0 / axvid1.fps)
            while len(kf_ts) > 2 and vid1_ofs_ms > kf_ts[1][0]:
                kf_ts = kf_ts[1:]
                kf_poly_coefs = kf_poly_coefs[1:]
            a = kf_ts[0][0]
            b = kf_ts[1][0]
            if vid1_ofs_ms < a:
                continue
            percentage = (vid1_ofs_ms - a) / (b - a)
            a_poly_val = poly.polyval(vid1_ofs_ms, kf_poly_coefs[0])
            b_poly_val = poly.polyval(vid1_ofs_ms, kf_poly_coefs[1])
            time_slip_data.append((a_poly_val * (1 - percentage)) + (b_poly_val * percentage) - vid1_ofs_ms)
    
    min_time_slip = min(time_slip_data)
    time_slip_range = max(time_slip_data) - min_time_slip
    image_height = int(num_vid1_frames*TIME_SLIP_GRAPH_HEIGHT/1920)
    time_slip_graph_image = Image.new('RGB', (num_vid1_frames,image_height), (0,0,0))
    draw = ImageDraw.Draw(time_slip_graph_image)
    line_points = [(frame, image_height - int(1 + (image_height - 2) * ((time_slip - min_time_slip) / time_slip_range))) for frame, time_slip in enumerate(time_slip_data)]
    draw.line(line_points, width=3)
    np_frame = np.array(list(time_slip_graph_image.getdata())).reshape(image_height,num_vid1_frames,3).astype('float32')
    return np.uint8(cv2.resize(np_frame, (1920, TIME_SLIP_GRAPH_HEIGHT), interpolation=cv2.INTER_AREA))

def draw_time_slip_frame(out_vid_frame_queue, prev_frame_done_event, this_frame_done_event,
                         frame1, frame2, vid1_elapsed, vid2_elapsed,
                         total_time_slip, time_slip_since_last_frame,
                         overlay_cm, graph, progress):
    if not hasattr(draw_time_slip_frame, "cached_font"):
        draw_time_slip_frame.cached_font = ImageFont.truetype(os.path.join(paths['FONTS'], "VollkornRegular.ttf"), size=40)

    out_frame = np.zeros((1080,1920,3),dtype=np.uint8)
    out_frame[:540,:960,:]  = cv2.resize(frame1, (960,540))
    out_frame[:540,-960:,:] = cv2.resize(frame2, (960,540))

    slip_change_magnitude = min(1, abs(time_slip_since_last_frame) * 60 / 100)
    v = 80 # percentage
    s = int(min(100, slip_change_magnitude * 100))
    if time_slip_since_last_frame < 0:
        h = 120
    else:
        h = 240
    # print(time_slip_since_last_frame, (h, s, v))
    bar_color = ImageColor.getrgb("hsv(%d,%d%%,%d%%)" % (h,s,v))
    if total_time_slip < 0:
        font_color = (50,50,255)
    elif total_time_slip > 0:
        font_color = (50,255,50)
    else:
        font_color = (200,200,200)

    time_slip_bar_image = Image.new('RGB', (1920, TIMING_HEIGHT + SLIP_BAR_HEIGHT + SLIP_TEXT_HEIGHT), (0,0,0))
    draw = ImageDraw.Draw(time_slip_bar_image)

    draw.text((280,TIMING_Y), "Elapsed: %0.3f (%.3f)" % (vid1_elapsed / 1000.0, -total_time_slip / 1000.0),
               font=draw_time_slip_frame.cached_font, fill=(255,255,255))
    draw.text((1240,TIMING_Y), "Elapsed: %0.3f (%.3f)" % (vid2_elapsed / 1000.0, total_time_slip / 1000.0),
               font=draw_time_slip_frame.cached_font, fill=(255,255,255))

    time_delta_mag = min(960, 960 * abs(total_time_slip/4000))
    text_x = 960
    if total_time_slip < 0:
        text_x = 960 - time_delta_mag
        draw.rectangle([960, SLIP_BAR_Y, 960+time_delta_mag, SLIP_TEXT_Y], fill=bar_color, width=0)
    elif total_time_slip > 0:
        text_x = 960 + time_delta_mag
        draw.rectangle([960-time_delta_mag, SLIP_BAR_Y, 960, SLIP_TEXT_Y], fill=bar_color, width=0)
    draw.text((text_x,SLIP_TEXT_Y - 10), "%+.3f" % (-total_time_slip / 1000.0), font=draw_time_slip_frame.cached_font, fill=font_color)
    time_slip_bar_np = np.array(list(time_slip_bar_image.getdata())).reshape(TIMING_HEIGHT + SLIP_BAR_HEIGHT + SLIP_TEXT_HEIGHT,1920,3)
    time_slip_bar_np = np.uint8(time_slip_bar_np)
    # print(time_slip_bar_np.shape, np.hstack((frame1_np, frame2_np)).shape)

    out_frame[540:540+TIMING_HEIGHT + SLIP_BAR_HEIGHT + SLIP_TEXT_HEIGHT,:,:] = time_slip_bar_np
    #frame_np = np.vstack((np.hstack((frame1_np, frame2_np)),time_slip_bar_np))
    if graph is not None and progress is not None:
        time_delta_mag = min(960, 960 * abs(total_time_slip/4000))
        #frame_np = np.vstack((frame_np, graph))
        out_frame[-TIME_SLIP_GRAPH_HEIGHT:,:,:] = graph
        play_head_center = int(1920 * progress)
        out_frame[-TIME_SLIP_GRAPH_HEIGHT:,
                 max(0,play_head_center - 1):min(1919,play_head_center + 1),
                 :] = 255
    
    if OVERLAY_COST_MATRIX_VISUALIZATION and overlay_cm is not None:
        out_frame[:COST_MATRIX_OVERLAY_SIZE,
                 960-int(COST_MATRIX_OVERLAY_SIZE/2):960+int(COST_MATRIX_OVERLAY_SIZE/2)] = cv2.cvtColor(overlay_cm, cv2.COLOR_GRAY2RGB)
    
    prev_frame_done_event.wait()
    out_vid_frame_queue.put(out_frame)
    this_frame_done_event.set()

def vid_read_worker(frame_queue, done_event, axvid, start_ms, start_frame):
    vid_player = ax_vid_video.VidPlayer(axvid)
    vid_player.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur_frame_num = start_frame
    cur_frame = None
    cur_ts = None
    while not done_event.is_set():
        if cur_frame is None:
            cur_ts = vid_player.get(cv2.CAP_PROP_POS_MSEC) - start_ms
            cur_frame_num += 1
            success, cur_frame = vid_player.read()
            if not success:
                break
        try:
            frame_queue.put((cur_frame_num - 1, cur_ts, cur_frame), timeout=0.25)
            cur_frame = None
        except Full:
            pass

def vid_write_worker(frame_queue, done_event, filename, fps):
    vid_writer = cv2.VideoWriter(filename,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps,
                                 (1920,1080))
    while not frame_queue.empty() or not done_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.25)
            vid_writer.write(frame)
        except Empty:
            pass
    vid_writer.release()

def create_comparison_video(filename, axvid1, axvid2, initial_cost_matrix):
    vid1_player = ax_vid_video.VidPlayer(axvid1)
    vid2_player = ax_vid_video.VidPlayer(axvid2)
    output_fps = min(OUTPUT_FPS, axvid1.fps)
    # Set play head to 1 frame prior to the desired start frame
    vid1_player.set(cv2.CAP_PROP_POS_MSEC, max(0, axvid1.StartInMS() - (1000.0 / axvid1.fps)))
    vid2_player.set(cv2.CAP_PROP_POS_MSEC, max(0, axvid2.StartInMS() - (1000.0 / axvid2.fps)))
    # Read one frame to align player to desired start frame
    vid1_player.read()
    vid2_player.read()
    vid1_start_frame = vid1_player.get(cv2.CAP_PROP_POS_FRAMES)
    vid2_start_frame = vid2_player.get(cv2.CAP_PROP_POS_FRAMES)
    vid1_start_ms = vid1_player.get(cv2.CAP_PROP_POS_MSEC)
    vid2_start_ms = vid2_player.get(cv2.CAP_PROP_POS_MSEC)
    cur_vid1_ts = vid1_start_ms
    cur_vid2_ts = vid2_start_ms
    font = ImageFont.truetype(os.path.join(paths['FONTS'], "VollkornRegular.ttf"), size=40)
    
    key_frames = identify_key_frames(initial_cost_matrix)
    vid1_len_ms = (float(key_frames[-1][0]) / DATA_HZ) * 1000.0
    ax_vid_progress.StartProgress('Creating video',
                                  'Creating time slip video...',
                                  vid1_len_ms)
    bar = ProgressBar(widgets=['Creating time slip video ',
                               Percentage(), ' ',
                               Bar(marker=RotatingMarker()), ' ',
                               ETA()], maxval=vid1_len_ms).start()
    
    if OVERLAY_COST_MATRIX_VISUALIZATION:
        def invert_func(num):
            return 255 - num
        invert = np.vectorize(invert_func)
        pretty_matrix = invert(initial_cost_matrix).astype(np.uint8)

    if USE_POLY_KEY_FRAME_SMOOTHING:
        #vid_writer = cv2.VideoWriter(filename,
        #                            cv2.VideoWriter_fourcc(*'mp4v'),
        #                            axvid1.fps,
        #                            (1920,1080))
        kf_ts, kf_poly_coefs = get_key_frame_poly_data(key_frames)


        # It turns out all this multithreading was worhtless, because Python
        # apparently can't run two threads at once due to GIL. You can use the
        # multiprocessing library to get around it, but it's too much work to
        # synchronize them for me to deem it worth it...
        done_event = Event()

        out_vid_frame_queue = Queue(maxsize=os.cpu_count()) 
        vid_write_thread = Thread(target=vid_write_worker, args=(out_vid_frame_queue, done_event, filename, axvid1.fps))

        vid1_frame_queue = Queue(maxsize=os.cpu_count())
        vid1_read_thread = Thread(target=vid_read_worker, args=(vid1_frame_queue, done_event, axvid1, vid1_start_ms, vid1_start_frame))

        vid2_frame_queue = Queue(maxsize=os.cpu_count())
        vid2_read_thread = Thread(target=vid_read_worker, args=(vid2_frame_queue, done_event, axvid2, vid2_start_ms, vid2_start_frame))

        vid_write_thread.start()
        vid1_read_thread.start()
        vid2_read_thread.start()

        #success2, frame2 = vid2_player.read()
        (vid2_cur_frame, vid2_ofs_ms, frame2) = vid2_frame_queue.get()
        last_frame_time_slip_ms = 0
        graph = get_time_slip_graph(axvid1, key_frames)
        prev_frame_done_event = Event()
        prev_frame_done_event.set()
        while True:
            (vid1_cur_frame, vid1_ofs_ms, frame1) = vid1_frame_queue.get()
            #vid1_ofs_ms = vid1_player.get(cv2.CAP_PROP_POS_MSEC) - vid1_start_ms
            #vid2_ofs_ms = vid2_player.get(cv2.CAP_PROP_POS_MSEC) - vid2_start_ms
            #success1, frame1 = vid1_player.read()
            while len(kf_ts) > 2 and vid1_ofs_ms > kf_ts[1][0]:
                kf_ts = kf_ts[1:]
                kf_poly_coefs = kf_poly_coefs[1:]
            a = kf_ts[0][0]
            b = kf_ts[1][0]
            #if not success1 or not success2 or vid1_ofs_ms > b or vid1_ofs_ms > vid1_len_ms:
            if vid1_ofs_ms > b or vid1_ofs_ms > vid1_len_ms:
                break
            if vid1_ofs_ms < a:
                continue
            # if vid1_ofs_ms > 10000:
            #     break
            ax_vid_progress.UpdateProgress(vid1_ofs_ms)
            bar.update(vid1_ofs_ms)
            percentage = (vid1_ofs_ms - a) / (b - a)
            a_poly_val = poly.polyval(vid1_ofs_ms, kf_poly_coefs[0])
            b_poly_val = poly.polyval(vid1_ofs_ms, kf_poly_coefs[1])
            target_vid2_ofs_ms = (a_poly_val * (1 - percentage)) + (b_poly_val * percentage)
            while vid2_ofs_ms < (target_vid2_ofs_ms - (0.5 * (1 / DATA_HZ))):
                (vid2_cur_frame, vid2_ofs_ms, frame2) = vid2_frame_queue.get()
            #while success2 and vid2_ofs_ms < target_vid2_ofs_ms:
                #success2, frame2 = vid2_player.read()
                #vid2_ofs_ms = vid2_player.get(cv2.CAP_PROP_POS_MSEC) - vid2_start_ms
            #if not success1 or not success2:
            #    break
            this_frame_time_slip_ms = target_vid2_ofs_ms - vid1_ofs_ms
            overlay_cm = None
            if OVERLAY_COST_MATRIX_VISUALIZATION:
                center_x = int((vid1_cur_frame - vid1_start_frame) / (axvid1.fps / DATA_HZ))
                center_y = int((vid2_cur_frame - vid2_start_frame) / (axvid2.fps / DATA_HZ))
                if center_x < pretty_matrix.shape[0] and center_y < pretty_matrix.shape[1]:
                    pretty_matrix[center_x][center_y] = 255
                win_start_x = min(max(0, center_x - int(COST_MATRIX_OVERLAY_SIZE / 2)),
                                  pretty_matrix.shape[0] - COST_MATRIX_OVERLAY_SIZE)
                win_start_y = min(max(0, center_y - int(COST_MATRIX_OVERLAY_SIZE / 2)),
                                  pretty_matrix.shape[1] - COST_MATRIX_OVERLAY_SIZE)
                overlay_cm = pretty_matrix[win_start_x:win_start_x + COST_MATRIX_OVERLAY_SIZE,
                                           win_start_y:win_start_y + COST_MATRIX_OVERLAY_SIZE]
            this_frame_done_event = Event()
            Thread(target=draw_time_slip_frame, args=(out_vid_frame_queue, prev_frame_done_event, this_frame_done_event,
                                                      frame1, frame2, vid1_ofs_ms, vid2_ofs_ms,
                                                      this_frame_time_slip_ms,
                                                      last_frame_time_slip_ms - this_frame_time_slip_ms,
                                                      overlay_cm, graph, vid1_ofs_ms / vid1_len_ms)).start()
            prev_frame_done_event = this_frame_done_event
            last_frame_time_slip_ms = this_frame_time_slip_ms

        # Make sure all the frames are in the queue before signalling completion
        this_frame_done_event.wait()

        done_event.set()
        vid_write_thread.join()
        vid1_read_thread.join()
        vid2_read_thread.join()
    else:
        vid_writer = cv2.VideoWriter(filename,
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    output_fps,
                                    (1920,640))
        vid1_ts_delta = 1000.0/output_fps #fps1
        for kf, (vid1_kf_index, vid2_kf_index) in enumerate(key_frames):
            vid1_kf_ts = vid1_start_ms + ((float(vid1_kf_index) / DATA_HZ) * 1000.0)
            vid2_kf_ts = vid2_start_ms + ((float(vid2_kf_index) / DATA_HZ) * 1000.0)

            num_frames_till_next_kf = (vid1_kf_ts - cur_vid1_ts) / vid1_ts_delta
            vid2_ts_delta = (vid2_kf_ts - cur_vid2_ts) / num_frames_till_next_kf

            time_slip = (vid1_ts_delta - vid2_ts_delta) * 1000.0
            time_delta = ((cur_vid1_ts - vid1_start_ms) - (cur_vid2_ts - vid2_start_ms))
            if time_slip < 0:
                bar_color = (50,50,255)
            else:
                bar_color = (50,255,50)
            if kf == 0:
                font_color = bar_color
            elif time_delta < 0:
                font_color = (50,50,255)
            elif time_delta > 0:
                font_color = (50,255,50)
            else:
                font_color = (200,200,200)

            while cur_vid1_ts < vid1_kf_ts:
                bar.update(cur_vid1_ts - vid1_start_ms)
                vid1_player.set(cv2.CAP_PROP_POS_MSEC, cur_vid1_ts)
                vid2_player.set(cv2.CAP_PROP_POS_MSEC, cur_vid2_ts)
                success1,frame1 = vid1_player.read()
                success2,frame2 = vid2_player.read()
                if not success1 or not success2:
                    print("failed to read frame (%d,%d,%d,%d)" % (bool(success1),bool(success2),cur_vid1_ts,cur_vid2_ts))
                    exit()
                cur_vid1_ts += vid1_ts_delta
                cur_vid2_ts += vid2_ts_delta

                frame1_np = cv2.resize(frame1, (960,540))
                frame2_np = cv2.resize(frame2, (960,540))

                time_slip_bar_image = Image.new('RGB', (1920,100), (0,0,0))
                draw = ImageDraw.Draw(time_slip_bar_image)
                # print("%.3f" % (((cur_vid1_ts - vid1_start_ms) - (cur_vid2_ts - vid2_start_ms)) / 1000.0))
                time_delta = ((cur_vid1_ts - vid1_start_ms) - (cur_vid2_ts - vid2_start_ms))
                time_delta_mag = min(960, 960 * abs(time_delta/4000))
                text_x = 960
                if time_delta < 0:
                    text_x = 960 - time_delta_mag
                    draw.rectangle([960-time_delta_mag, 0, 960, 50], fill=bar_color, width=0)
                elif time_delta > 0:
                    text_x = 960 + time_delta_mag
                    draw.rectangle([960, 0, 960+time_delta_mag, 50], fill=bar_color, width=0)
                draw.text((text_x,40), "%.3f" % (((cur_vid1_ts - vid1_start_ms) - (cur_vid2_ts - vid2_start_ms)) / 1000.0),
                        font=font, fill=font_color)
                time_slip_bar_np = np.array(list(time_slip_bar_image.getdata())).reshape(100,1920,3)
                time_slip_bar_np = np.uint8(time_slip_bar_np)
                # print(time_slip_bar_np.shape, np.hstack((frame1_np, frame2_np)).shape)

                vid_writer.write(np.vstack((np.hstack((frame1_np, frame2_np)),time_slip_bar_np)))
    
    ax_vid_progress.EndProgress()
    bar.finish()
    return vid1_ofs_ms / 1000.0

def GenerateWithProcessedVids(vid1, vid2):
    (axvid1, vid1_player, vid1_data) = vid1
    (axvid2, vid2_player, vid2_data) = vid2
    vid1_id = axvid1.yt_id
    vid2_id = axvid2.yt_id

    cm_file_base = os.path.join(paths['COST_MATRICES'], "%s_%s_%dhz_" % (vid1_id[:3], vid2_id[:3], DATA_HZ))

    cost_matrix = None
    cached_matrix_loaded = False
    if LOAD_CACHED_COST_MATRIX:
        cached_cm_file_name = cm_file_base + "compare_matrix_raw.png"
        if os.path.exists(cached_cm_file_name):
            def invert_func(num):
                return 255 - num
            invert = np.vectorize(invert_func)
            cached_cm = cv2.imread(cached_cm_file_name, flags=cv2.IMREAD_GRAYSCALE)
            cost_matrix = invert(cached_cm.astype('float32'))
            cached_matrix_loaded = True
    
    if cost_matrix is None:
        cost_matrix = get_initial_cost_matrix(vid1_id, vid1_data, vid2_id, vid2_data)
    
    if OUTPUT_COST_MATRICES:
        def invert_func(num):
            return 255 - num
        invert = np.vectorize(invert_func)

        alignment = dtw.dtw(cost_matrix, step_pattern='typeIbs')

        path = list(zip(alignment.index1, alignment.index2))
        smoothed_path = []
        smothing_window = DATA_HZ
        for i in range(len(path)):
            start = max(0, i - int(smothing_window / 2))
            end = min(len(path) - 1, start + smothing_window)
            i1_avg = int(np.mean([i1 for (i1,_) in path[start:end]]))
            i2_avg = int(np.mean([i2 for (_,i2) in path[start:end]]))
            smoothed_path.append((i1_avg,i2_avg))

        pretty_matrix = invert(cost_matrix)

        if not cached_matrix_loaded:
            if ONLY_FIRST_N != np.inf:
                print("WARNING: NOT saving incomplete cached cost matrix")
            else:
                cv2.imwrite(cm_file_base + "compare_matrix_raw.png", pretty_matrix)

        visualize_path = np.zeros((cost_matrix.shape[0], 20))
        for (x,y) in smoothed_path:
            visualize_path[x] = np.full((20), pretty_matrix[x][y])
            pretty_matrix[x][y] = 255
        
        key_frames = identify_key_frames(cost_matrix)
        for (i1,i2) in key_frames:
            for (x,y) in zip(range(i1 + 5, i1 - 6, -1), range(i2 - 5, i2 + 6)):
                try:
                    pretty_matrix[x][y] = 255
                except:
                    pass
        cv2.imwrite(cm_file_base + "compare_matrix.png", pretty_matrix)
        
        if USE_POLY_KEY_FRAME_SMOOTHING:
            poly_matrix = invert(cost_matrix)
            for (i1,i2) in key_frames:
                for (x,y) in zip(range(i1 + 5, i1 - 6, -1), range(i2 - 5, i2 + 6)):
                    try:
                        poly_matrix[x][y] = 255
                    except:
                        pass
            kf_ts, kf_poly_coefs = get_key_frame_poly_data(key_frames)
            for i1 in range(poly_matrix.shape[0]):
                vid1_ofs_ms = (i1 * (1 / DATA_HZ)) * 1000.0
                while len(kf_ts) > 2 and vid1_ofs_ms > kf_ts[1][0]:
                    kf_ts = kf_ts[1:]
                    kf_poly_coefs = kf_poly_coefs[1:]
                a = kf_ts[0][0]
                b = kf_ts[1][0]
                if vid1_ofs_ms < a:
                    continue
                percentage = (vid1_ofs_ms - a) / (b - a)
                a_poly_val = poly.polyval(vid1_ofs_ms, kf_poly_coefs[0])
                b_poly_val = poly.polyval(vid1_ofs_ms, kf_poly_coefs[1])
                target_vid2_ofs_ms = (a_poly_val * (1 - percentage)) + (b_poly_val * percentage)

                ya = int(a_poly_val / 1000.0 * DATA_HZ)
                yb = int(b_poly_val / 1000.0 * DATA_HZ)
                yres = int(target_vid2_ofs_ms / 1000.0 * DATA_HZ)
                try:
                    # poly_matrix[i1][ya] = 255
                    # poly_matrix[i1][yb] = 255
                    poly_matrix[i1][yres] = 255
                except:
                    pass
            cv2.imwrite(cm_file_base + "compare_matrix_poly.png", poly_matrix)

        # cv2.imwrite(cm_file_base + "path_viz.png", visualize_path)
    
    if OUTPUT_TIME_SLIP_VIDEO:
        filepath = os.path.join(paths['OUTPUT_VIDEOS'], vid1_id + "_" + vid2_id + ".mp4")
        length_s = create_comparison_video(filepath, axvid1, axvid2, cost_matrix)
        if True:
            vid = mpe.VideoFileClip(filepath)
            aud = mpe.AudioFileClip(ax_vid_video.AudFilename(axvid1))
            start_s = axvid1.StartInMS() / 1000.0
            aud = aud.subclip(start_s, start_s + length_s)
            #aud.write_audiofile(filepath[:-4] + "_sound.m4a")
            vid.set_audio(aud).write_videofile(filepath[:-4] + "_sound.mp4", audio_codec="aac", fps=axvid1.fps)
        else:
            input_vid = ffmpeg.input(filepath)
            input_aud = ffmpeg.input(ax_vid_video.AudFilename(axvid1),
                                    ss='%f'%(axvid1.StartInMS()/1000.0),
                                    t='%f'%(length_s))
            sound_filepath = filepath[:-4] + "_sound.mp4"
            ffmpeg.concat(input_vid, input_aud, v=1, a=1).output(sound_filepath).overwrite_output().run(quiet=False)

   
def DoVidCompare(axvid1, axvid2):
    GenerateWithProcessedVids(process_video(axvid1), process_video(axvid2))
     
if __name__ == '__main__':
    cpuinfo.get_cpu_info()
    print("Must be invoked via ax_vid_gui.py")