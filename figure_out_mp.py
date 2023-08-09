import os
import multiprocessing as mp
from multiprocessing.pool import Pool
from queue import Empty, Full

def process_func(done_event, queue):
    while not done_event.is_set() or not queue.empty():
        try:
            str = queue.get(timeout=0.25)
            print(str, flush=True)
        except Empty:
            pass

if __name__ == '__main__':
    done_event = mp.Event()
    draw_frame_args_queue = mp.Queue(maxsize=os.cpu_count())
    next_frame = mp.Value('L') # 'L' for unsigned long
    next_frame.value = 0
    draw_frame_worker_pool = Pool(4)
    proc = mp.Process(target=process_func, args=(done_event, draw_frame_args_queue))
    proc.start()

    out_vid_frame_queue = mp.Queue(maxsize=os.cpu_count())

    #draw_frame_worker_pool.apply_async(func=process_func, args=(done_event, draw_frame_args_queue))

    [draw_frame_args_queue.put(str(x)) for x in range(10)]

    done_event.set()
    proc.join()