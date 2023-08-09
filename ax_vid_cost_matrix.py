import os
import cv2
import icp
import math
import numpy as np
import numpy.polynomial.polynomial as poly
import dtw
from ax_vid_files import paths
from ax_vid_config import *
import ax_vid_progress

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

def get_initial_cost_matrix(vid1_data, vid2_data):
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

def GetCostMatrix(vid1_data, vid2_data, pair_id):
    cm_file_base = os.path.join(paths['COST_MATRICES'], "%s_%dhz_" % (pair_id, DATA_HZ))

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
        cost_matrix = get_initial_cost_matrix(vid1_data, vid2_data)
    
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
    return cost_matrix