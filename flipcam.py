import numpy as np
import cv2
import time
import argparse
import threading
import queue

parser = argparse.ArgumentParser(description='Simple demo for WebCam flipping')
parser.add_argument('-i', '--interval', default=30, type=float,
                    help='Minimal interval between frame, '
                    'in milliseconds, default is 50')
parser.add_argument('-c', '--camera', default=0, type=int,
                    help='Camera ID, an integer, '
                    'default is 0')
parser.add_argument('-q', '--quit-button', default='q',
                    help='Button to quit the window, '
                    'in lowercase, default is q')

parser.add_argument('-p', '--pct', default=0.5, type=float,
                    help='PCT')

FLAGS = parser.parse_args()


class StreamCameraReader:
    def __init__(self, camera_id):
        self._stream_buffer = queue.LifoQueue(1)
        self._cap = cv2.VideoCapture(camera_id)

        self.stop_flag = False

    def start(self):
        def read_frames():
            succeed = True
            while not self.stop_flag and succeed:
                succeed, frame = self._cap.read()
                if self._stream_buffer.full():
                    self._stream_buffer.get()
                self._stream_buffer.put(frame)

        worker = threading.Thread(target=read_frames)
        worker.start()

    def get_frame(self):
        return self._stream_buffer.get()


dt0, dt1, dt2 = 0, 0, 0
i = 0

fps = 0

camera_reader = StreamCameraReader(FLAGS.camera)
camera_reader.start()

while True:
    t_epoch = time.time()
    t0 = time.time()
    # Capture frame-by-frame
    frame = camera_reader.get_frame()
    dt0 += time.time() - t0
    t0 = time.time()

    # downsampling
    H0, W0, _ = frame.shape
    new_size = (int(W0 * FLAGS.pct), int(H0 * FLAGS.pct))
    frame = cv2.resize(frame, new_size)

    # flip the first dimension
    frame_f = frame[::-1, ::-1, :]

    dt1 += time.time() - t0
    t0 = time.time()

    # Display the resulting frame
    cv2.imshow('WebCam Stream', frame_f)

    dt2 += time.time() - t0
    t0 = time.time()

    dt = time.time() - t_epoch

    residual_interval_ms = int(max(FLAGS.interval - dt * 1000, 1))

    fps = 1 / (residual_interval_ms / 1000 + dt)

    i += 1
    key = cv2.waitKey(residual_interval_ms) & 0xFF

    if key == ord('q'):
        camera_reader.stop_flag = True
        break

    print('FPS=%.2f' % fps)

print('Report of time cost')
print('Frame capture: %.2fms' % (dt0 / i * 1000))
print('Fliping: %.2fms' % (dt1 / i * 1000))
print('Frame display %.2fms' % (dt2 / i * 1000))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
