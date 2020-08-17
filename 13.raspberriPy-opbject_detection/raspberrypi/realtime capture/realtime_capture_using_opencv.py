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




def label_image(image_data, sess):
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')  # final_result:0 is Tensor
        image_data = cv2.normalize(image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        print(type(image_data))
        print("TIME CHECK start")
        timeCheck()

        predictions = sess.run(softmax_tensor, \
                               {'Mul:0': image_data})

        print("TIME CHECK end")
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
        try:
            for f in self.camera.capture_continuous(self.rawCapture,
                                                    format="bgr",
                                                    use_video_port=True):
                # NumPy array로 받음
                frame = f.array
                cv2.imshow("Feed", frame)
                image_data = np.asarray(frame)
                image_data = np.expand_dims(image_data, axis=0)
                label_image(image_data, sess)
                time.sleep(0)


                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                self.rawCapture.truncate(0)
        except KeyboardInterrupt: # must Ctrl + C
            self.stop()
            sys.exit()

    def stop(self):
        self.camera.close()


if __name__ == "__main__":
    c = NScapture(video_show=True)
    with tf.Session() as sess:
        c.run(sess)
    c.stop()