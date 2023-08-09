import os
import cv2
import time
import numpy as np
import numpy.polynomial.polynomial as poly
import moviepy.editor as mpe
#from multiprocessing import Queue, Value, Event, set_start_method
#from threading import Thread as Process
from threading import Thread
from multiprocessing import Process, Queue, Value, Event, set_start_method
from multiprocessing.shared_memory import SharedMemory
from queue import Full, Empty
from PIL import Image, ImageDraw, ImageFont, ImageColor
from ax_vid_files import paths
from ax_vid_config import *
import ax_vid_cost_matrix as axv_cm
import ax_vid_video as axv_vid
import ax_vid_progress as axv_prog
import ax_vid_files as axv_files

#VIDEO DIMENSIONS
TIMING_HEIGHT = 50 # "Elapsed time: ##.##s (+/-#.###s)" just below the video
SLIP_BAR_HEIGHT = 50 # Green/Red time slip bar
SLIP_TEXT_HEIGHT = 50 # Green/Red time slip text below the bar
TIME_SLIP_GRAPH_HEIGHT = (1080 - 540 - TIMING_HEIGHT - SLIP_BAR_HEIGHT - SLIP_TEXT_HEIGHT) # Calculated to make a 1080p video
COST_MATRIX_OVERLAY_SIZE = 300 # pixels

TIMING_Y = 0 # Offset from base of middle image
SLIP_BAR_Y = TIMING_Y + TIMING_HEIGHT
SLIP_TEXT_Y = SLIP_BAR_Y + SLIP_BAR_HEIGHT
SLIP_GRAPH_Y = SLIP_TEXT_Y + SLIP_TEXT_HEIGHT

def get_time_slip_graph(axvid1, key_frames):
    num_vid1_frames = key_frames[-1][0]
    time_slip_data = []
    if USE_POLY_KEY_FRAME_SMOOTHING:
        kf_ts, kf_poly_coefs = axv_cm.get_key_frame_poly_data(key_frames)
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

def draw_frame(out_frame, frame1, frame2, vid1_elapsed, vid2_elapsed,
               total_time_slip, time_slip_since_last_frame,
               overlay_cm, graph, progress):
    if not hasattr(draw_frame, "cached_font"):
        draw_frame.cached_font = ImageFont.truetype(os.path.join(paths['FONTS'], "VollkornRegular.ttf"), size=40)

    #out_frame = np.zeros((1080,1920,3),dtype=np.uint8)
    out_frame[:540,:960,:]  = frame1 #cv2.resize(frame1, (960,540))
    out_frame[:540,-960:,:] = frame2 #cv2.resize(frame2, (960,540))

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
               font=draw_frame.cached_font, fill=(255,255,255))
    draw.text((1240,TIMING_Y), "Elapsed: %0.3f (%.3f)" % (vid2_elapsed / 1000.0, total_time_slip / 1000.0),
               font=draw_frame.cached_font, fill=(255,255,255))

    time_delta_mag = min(960, 960 * abs(total_time_slip/4000))
    text_x = 960
    if total_time_slip < 0:
        text_x = 960 - time_delta_mag
        draw.rectangle([960, SLIP_BAR_Y, 960+time_delta_mag, SLIP_TEXT_Y], fill=bar_color, width=0)
    elif total_time_slip > 0:
        text_x = 960 + time_delta_mag
        draw.rectangle([960-time_delta_mag, SLIP_BAR_Y, 960, SLIP_TEXT_Y], fill=bar_color, width=0)
    draw.text((text_x,SLIP_TEXT_Y - 10), "%+.3f" % (-total_time_slip / 1000.0), font=draw_frame.cached_font, fill=font_color)
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
    
    #if OVERLAY_COST_MATRIX_VISUALIZATION and overlay_cm is not None:
    #    out_frame[:COST_MATRIX_OVERLAY_SIZE,
    #             960-int(COST_MATRIX_OVERLAY_SIZE/2):960+int(COST_MATRIX_OVERLAY_SIZE/2)] = cv2.cvtColor(overlay_cm, cv2.COLOR_GRAY2RGB)

    return out_frame

def open_shared(num_workers):
    fbs_shm = SharedMemory(name="worker_fbs")
    fbs = np.ndarray((num_workers, 1080, 1920, 3), dtype=np.uint8, buffer=fbs_shm.buf)
    framenums_shm = SharedMemory(name="worker_framenums")
    framenums = np.ndarray((num_workers,), dtype=np.int32, buffer=framenums_shm.buf)
    return (fbs, framenums, fbs_shm, framenums_shm)

def draw_frame_worker(num_workers, draw_frame_args_queue, done_event, done_frame_num, init_counter):
    with init_counter.get_lock():
        init_counter.value += 1
    (fbs, framenums, fbs_shm, framenums_shm) = open_shared(num_workers)
    while not done_event.is_set() or not draw_frame_args_queue.empty():
        try:
            (frame_num, args) = draw_frame_args_queue.get(timeout=0.25)
            frame_index = frame_num % num_workers
            fb = fbs[frame_index]
            while (done_frame_num.value + num_workers) < frame_num:
                #print("waiting for %d < %d" % (frames_done.value + num_workers, frame_num), flush=True)
                time.sleep(0.01)
            #print("drawing fbs[%d]" % (frame_index,), flush=True)
            draw_frame(fb, *args)
            framenums[frame_index] = frame_num
            #print("framenums[%d] = %d" % (frame_index, frame_num), flush=True)
            #while next_frame.value != frame_num:
            #    time.sleep(0.01)
            #out_vid_frame_queue.put(out_frame)
            #with next_frame.get_lock():
            #    next_frame.value += 1
        except Empty:
            pass
    #out_vid_frame_queue.close()
    draw_frame_args_queue.close()
    fbs_shm.close()
    framenums_shm.close()

def vid_read_worker(frame_queue, done_event, axvid, start_ms, start_frame):
    vid_player = axv_vid.VidPlayer(axvid)
    vid_player.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cur_frame_num = start_frame
    cur_frame = None
    cur_ts = None
    while not done_event.is_set():
        if cur_frame is None:
            cur_ts = vid_player.get(cv2.CAP_PROP_POS_MSEC) - start_ms
            cur_frame_num += 1
            success, cur_frame = vid_player.read()
            cur_frame = cv2.resize(cur_frame, (960,540))
            if not success:
                break
        try:
            frame_queue.put((cur_frame_num - 1, cur_ts, cur_frame), timeout=0.25)
            cur_frame = None
        except Full:
            pass
    
    frame_queue.close()

def vid_write_worker(num_workers, done_frame_num, final_frame_num, filename, fps, init_event):
    init_event.set()
    #print("open_shared", flush=True)
    (fbs, framenums, fbs_shm, framenums_shm) = open_shared(num_workers)
    vid_writer = cv2.VideoWriter(filename,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps,
                                 (1920,1080))
    #print("loop %d" % (num_frames,), flush=True)
    while done_frame_num.value < final_frame_num.value:
        next_frame_num = done_frame_num.value + 1
        frame_index = (next_frame_num % num_workers)
        if framenums[frame_index] == next_frame_num:
            #print("writing fbs[%d]" % (frame_index,), flush=True)
            vid_writer.write(fbs[frame_index])
            with done_frame_num.get_lock():
                done_frame_num.value = next_frame_num
        else:
            #print("waiting for framenums[%d] == %d" % (frame_index, next_frame_num), flush=True)
            time.sleep(0.01)
    vid_writer.release()
    fbs_shm.close()
    framenums_shm.close()

def start_vid_out_worker(num_workers, filename, fps):
    init_event = Event()
    final_frame_num = Value('L')
    final_frame_num.value = 999999999
    done_frame_num = Value('l') # signed long
    done_frame_num.value = -1
    proc = Process(target=vid_write_worker, args=(num_workers, done_frame_num, final_frame_num, filename, fps, init_event))
    proc.start()
    init_event.wait()
    return (proc, final_frame_num, done_frame_num)

def start_drawing_workers(num_workers, done_frame_num):
    done_event = Event()
    draw_frame_args_queue = Queue(maxsize=1)
    init_counter = Value('L')
    init_counter.value = 0
    # Queue hilariously doesn't guarantee ordering, even if you use a
    # synchronization method to guarantee that items are added to the queue in
    # order. To work around this, set the queue size to 1.
    #out_vid_frame_queue = Queue(maxsize=1)

    procs = []
    for i in range(num_workers):
        procs.append(Process(target=draw_frame_worker,
                             args=(num_workers,
                                   draw_frame_args_queue,
                                   done_event,
                                   done_frame_num,
                                   init_counter)))
    [proc.start() for proc in procs]

    while init_counter.value != len(procs):
        time.sleep(.1)
    
    return (procs, done_event, draw_frame_args_queue)

def create_comparison_video(axvids, initial_cost_matrix):
    vid1_player = axv_vid.VidPlayer(axvids[0])
    vid2_player = axv_vid.VidPlayer(axvids[1])
    output_fps = min(OUTPUT_FPS, axvids[0].fps)
    # Set play head to 1 frame prior to the desired start frame
    vid1_player.set(cv2.CAP_PROP_POS_MSEC, max(0, axvids[0].StartInMS() - (1000.0 / axvids[0].fps)))
    vid2_player.set(cv2.CAP_PROP_POS_MSEC, max(0, axvids[1].StartInMS() - (1000.0 / axvids[1].fps)))
    # Read one frame to align player to desired start frame
    vid1_player.read()
    vid2_player.read()
    vid1_start_frame = vid1_player.get(cv2.CAP_PROP_POS_FRAMES)
    vid2_start_frame = vid2_player.get(cv2.CAP_PROP_POS_FRAMES)
    vid1_start_ms = vid1_player.get(cv2.CAP_PROP_POS_MSEC)
    vid2_start_ms = vid2_player.get(cv2.CAP_PROP_POS_MSEC)
    cur_vid1_ts = vid1_start_ms
    cur_vid2_ts = vid2_start_ms
    font = ImageFont.truetype(os.path.join(axv_files.paths['FONTS'], "VollkornRegular.ttf"), size=40)
    
    key_frames = axv_cm.identify_key_frames(initial_cost_matrix)
    vid1_len_ms = (float(key_frames[-1][0]) / DATA_HZ) * 1000.0
    axv_prog.StartProgress('Creating video',
                                  'Creating time slip video...',
                                  vid1_len_ms)
    
    if OVERLAY_COST_MATRIX_VISUALIZATION:
        def invert_func(num):
            return 255 - num
        invert = np.vectorize(invert_func)
        pretty_matrix = invert(initial_cost_matrix).astype(np.uint8)

    #vid_writer = cv2.VideoWriter(filename,
    #                            cv2.VideoWriter_fourcc(*'mp4v'),
    #                            axvid1.fps,
    #                            (1920,1080))
    kf_ts, kf_poly_coefs = axv_cm.get_key_frame_poly_data(key_frames)

    num_workers = max(1, os.cpu_count() - 2) # leave one cpu for main thread, one for vid writer
    fb_np = np.empty((num_workers,1080,1920,3),dtype=np.uint8)
    frame_num_np = np.empty((num_workers,),dtype=np.int32)
    framenums_shm = SharedMemory("worker_framenums", create=True, size=frame_num_np.nbytes)
    frame_num_np = np.ndarray((num_workers,), dtype=np.int32, buffer=framenums_shm.buf)
    frame_num_np[:] = -1
    fbs_shm = SharedMemory("worker_fbs", create=True, size=fb_np.nbytes)

    #(fbs, framenums) = axv_draw.open_shared(num_workers)
    
    (vid_writer_proc, final_frame_num, done_frame_num) = start_vid_out_worker(
        num_workers,
        axv_files.GetPairOutputFile(axvids),
        axvids[0].fps)
    (draw_frame_procs, draw_frame_done_event, draw_frame_args_queue) = start_drawing_workers(
        num_workers,
        done_frame_num)

    reader_done_event = Event()

    vid1_frame_queue = Queue(maxsize=2)
    vid1_read_thread = Thread(target=vid_read_worker, args=(vid1_frame_queue, reader_done_event, axvids[0], vid1_start_ms, vid1_start_frame))
    vid1_read_thread.start()

    vid2_frame_queue = Queue(maxsize=2)
    vid2_read_thread = Thread(target=vid_read_worker, args=(vid2_frame_queue, reader_done_event, axvids[1], vid2_start_ms, vid2_start_frame))
    vid2_read_thread.start()

    #success2, frame2 = vid2_player.read()
    (vid2_cur_frame, vid2_ofs_ms, frame2) = vid2_frame_queue.get()
    last_frame_time_slip_ms = 0
    graph = get_time_slip_graph(axvids[0], key_frames)
    frame_num = 0
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
        axv_prog.UpdateProgress(vid1_ofs_ms)
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
            center_x = int((vid1_cur_frame - vid1_start_frame) / (axvids[0].fps / DATA_HZ))
            center_y = int((vid2_cur_frame - vid2_start_frame) / (axvids[1].fps / DATA_HZ))
            if center_x < pretty_matrix.shape[0] and center_y < pretty_matrix.shape[1]:
                pretty_matrix[center_x][center_y] = 255
            win_start_x = min(max(0, center_x - int(COST_MATRIX_OVERLAY_SIZE / 2)),
                                pretty_matrix.shape[0] - COST_MATRIX_OVERLAY_SIZE)
            win_start_y = min(max(0, center_y - int(COST_MATRIX_OVERLAY_SIZE / 2)),
                                pretty_matrix.shape[1] - COST_MATRIX_OVERLAY_SIZE)
            overlay_cm = pretty_matrix[win_start_x:win_start_x + COST_MATRIX_OVERLAY_SIZE,
                                        win_start_y:win_start_y + COST_MATRIX_OVERLAY_SIZE]
        draw_frame_args_queue.put((frame_num, (frame1, frame2, vid1_ofs_ms, vid2_ofs_ms,
                                                this_frame_time_slip_ms,
                                                last_frame_time_slip_ms - this_frame_time_slip_ms,
                                                overlay_cm, graph, vid1_ofs_ms / vid1_len_ms)))
        frame_num += 1
        last_frame_time_slip_ms = this_frame_time_slip_ms

    reader_done_event.set()
    vid1_read_thread.join()
    vid2_read_thread.join()
    
    draw_frame_done_event.set()
    [proc.join() for proc in draw_frame_procs]
    draw_frame_args_queue.close()

    with final_frame_num.get_lock():
        final_frame_num.value = (frame_num - 1)
    vid_writer_proc.join()

    framenums_shm.close()
    framenums_shm.unlink()
    fbs_shm.close()
    fbs_shm.unlink()
    
    axv_prog.EndProgress()
    return vid1_ofs_ms / 1000.0


def GenerateWithProcessedVids(axvids, cm):
    if OUTPUT_TIME_SLIP_VIDEO:
        filepath = axv_files.GetPairOutputFile(axvids)
        length_s = create_comparison_video(axvids, cm)
        if True:
            vid = mpe.VideoFileClip(filepath)
            aud = mpe.AudioFileClip(axv_vid.AudFilename(axvids[0]))
            start_s = axvids[0].StartInMS() / 1000.0
            aud = aud.subclip(start_s, start_s + length_s)
            #aud.write_audiofile(filepath[:-4] + "_sound.m4a")
            vid.set_audio(aud).write_videofile(filepath[:-4] + "_sound.mp4", audio_codec="aac", fps=axvids[0].fps)
        else:
            input_vid = ffmpeg.input(filepath)
            input_aud = ffmpeg.input(ax_vid_video.AudFilename(axvid1),
                                    ss='%f'%(axvid1.StartInMS()/1000.0),
                                    t='%f'%(length_s))
            sound_filepath = filepath[:-4] + "_sound.mp4"
            ffmpeg.concat(input_vid, input_aud, v=1, a=1).output(sound_filepath).overwrite_output().run(quiet=False)