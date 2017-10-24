import cv2
import time
import threading
import queue
import fire


class StreamCameraReader:
    def __init__(self, camera_id):
        self._stream_buffer = queue.LifoQueue(1)
        self._camera_id = camera_id

        self.stop_flag = False

    def start(self):
        def read_frames():
            cap = cv2.VideoCapture(self._camera_id)
            succeed = True
            while not self.stop_flag and succeed:
                succeed, frame = cap.read()
                if self._stream_buffer.full():
                    self._stream_buffer.get()
                self._stream_buffer.put(frame)
            cap.release()

        self._worker = threading.Thread(target=read_frames)
        self._worker.start()

    def get_frame(self):
        return self._stream_buffer.get()

    def stop(self):
        self.stop_flag = True

    def join(self):
        self._worker.join()


def show_flipped(camera, interval, pct):
    camera = int(camera)
    interval = float(interval)
    pct = float(pct)

    camera_reader = StreamCameraReader(camera)
    camera_reader.start()

    dt0, dt1, dt2 = 0, 0, 0
    i = 0

    while True:
        t_epoch = time.time()
        t0 = time.time()
        # Capture frame-by-frame
        frame = camera_reader.get_frame()
        dt0 += time.time() - t0
        t0 = time.time()

        # downsampling
        H0, W0, _ = frame.shape
        new_size = (int(W0 * pct), int(H0 * pct))
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

        residual_interval_ms = int(max(interval - dt * 1000, 1))

        fps = 1 / (residual_interval_ms / 1000 + dt)

        i += 1
        key = cv2.waitKey(residual_interval_ms) & 0xFF

        if key == ord('q'):
            camera_reader.stop()
            break

        if key == ord('c'):
            cv2.imwrite('sample.jpg', frame)
            print('Image saved')

        print('FPS=%.2f' % fps)

    print('Report of time cost')
    print('Frame capture: %.2fms' % (dt0 / i * 1000))
    print('Fliping: %.2fms' % (dt1 / i * 1000))
    print('Frame display %.2fms' % (dt2 / i * 1000))

    cv2.destroyAllWindows()
    camera_reader.join()


class Main:
    def show_flipped(self, camera=0, interval=10, pct=0.5):
        # Show flipped camera stream
        # camera: Camera ID, an integer, default is 0
        # interval: Minimal interval between frame, in milliseconds,
        # default is 50
        # pct: Downsampling rate (for faster display speed)
        show_flipped(camera, interval, pct)

    def capture_sample(self, camera=0, output='sample.jpg'):
        # Capture an single image
        camera_reader = StreamCameraReader(camera)
        camera_reader.start()
        frame = camera_reader.get_frame()
        cv2.imwrite(output, frame)
        camera_reader.stop()
        camera_reader.join()


if __name__ == '__main__':
    fire.Fire(Main)
