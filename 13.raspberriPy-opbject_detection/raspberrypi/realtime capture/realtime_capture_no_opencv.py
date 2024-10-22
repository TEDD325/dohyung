from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import datetime
import tensorflow as tf
import time
import numpy as np
import sys

t0 = time.time()

def timeCheck():
    t1 = time.time()
    print("until now time :", t1 - t0)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in
               tf.gfile.GFile("/home/pi/raspberrypi_project/Testing_OnRaspberryPi/restoreOutput/output_labels.txt")]
# Unpersists graph from file
with tf.gfile.FastGFile("/home/pi/raspberrypi_project/Testing_OnRaspberryPi/restoreOutput/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


# about 6~10 second consumed


def label_image(image_data, sess):

        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')  # final_result:0 is Tensor
        image_data = cv2.normalize(image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        print(type(image_data))
        print("TIME CHECK--")
        timeCheck()

        predictions = sess.run(softmax_tensor, \
                               {'Mul:0': image_data})
        # about 6~7 second consumed

        print("TIME CHECK--")
        timeCheck()

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
        print()


class NScapture:
    conf = {
        # "SAVE_DIR": "/home/pi/tf_files/test",
        "resolution": (299, 299),
        "fps": 16,
        "warmup": 1.5,
        # "fname_prefix": 'NeuroSafe',
        "video_show": False
    }

    def __init__(self, **kwargs):
        if kwargs is not None:
            for k, v in kwargs.items():
                self.conf[k] = v

        self.camera = PiCamera()
        self.camera.resolution = self.conf["resolution"]
        self.camera.framerate = self.conf["fps"]
        self.camera.hflip = True
        self.rawCapture = PiRGBArray(self.camera, size=self.conf["resolution"])
        time.sleep(self.conf["warmup"])

    def run(self, sess):
        avg = None
        lastUploaded = datetime.datetime.now()
        motionCounter = 0
        minDelay = 1.0
        minFrame = 8
        delta_thresh = 5

        try:
            for f in self.camera.capture_continuous(self.rawCapture,
                                                    format="bgr",
                                                    use_video_port=True):
                # NumPy array로 받음
                frame = f.array
                timestamp = datetime.datetime.now()
                status = "Normal"
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Gaussian Smoothing
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if avg is None:
                    avg = gray.copy().astype("float")
                    self.rawCapture.truncate(0)
                    continue
                cv2.accumulateWeighted(gray, avg, 0.5)

                frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

                thresh = cv2.threshold(frameDelta,
                                       delta_thresh,
                                       255,
                                       cv2.THRESH_BINARY)[1]

                thresh = cv2.dilate(thresh, None, iterations=2)

                _, cnts, _ = cv2.findContours(thresh.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

                for c in cnts:
                    if cv2.contourArea(c) < 5000:
                        continue
                    status = "Capture"

                # ts = timestamp.strftime("%Y%m%d_%H%M%S_%f")
                # filename="{}-{}.jpg".format(self.conf["SAVE_DIR"]+self.conf["fname_prefix"],ts)

                if status == "Capture":
                    if (timestamp - lastUploaded).seconds >= minDelay:
                        motionCounter += 1
                        image_data = np.asarray(frame)
                        # about 5 second consumed

                        image_data = np.expand_dims(frame, axis=0)

                        if motionCounter >= minFrame:
                            label_image(image_data, sess)

                            # cv2.imwrite(("/home/pi/tf_files/test/"+filename),frame)
                            lastUploaded = timestamp
                            motionCounter = 0

                else:
                    motionCounter = 0

                if self.conf["video_show"]:
                    cv2.imshow("Feed", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                self.rawCapture.truncate(0)
        except KeyboardInterrupt:
            self.stop()
            sys.exit()

    def stop(self):
        self.camera.close()


if __name__ == "__main__":
    c = NScapture(video_show=True)
    with tf.Session() as sess:
        c.run(sess)
    c.stop()