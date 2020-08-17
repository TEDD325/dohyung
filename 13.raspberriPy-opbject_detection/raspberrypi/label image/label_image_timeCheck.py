import tensorflow as tf
import sys
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time
import datetime


t0 = time.time()

def timeCheck():
    t1 = time.time()
    print("until now time : %d", t1-t0)

# change this as you see fit
image_path = sys.argv[1]
image_path2 = sys.argv[2]
# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()
image_data2 = tf.gfile.FastGFile(image_path2, 'rb').read()
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("/home/pi/tf_files/output_labels.txt")]
# Unpersists graph from file

with tf.gfile.FastGFile("/home/pi/tf_files/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    timeCheck() #7 second



    predictions = sess.run(softmax_tensor, \
                           {'DecodeJpeg/contents:0': image_data})
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
    timeCheck() #15 second


    ##
    ## 하드디스크에서 image data 불러오는 코드
    ##
    ################################
    predictions = sess.run(softmax_tensor, \
                           {'DecodeJpeg/contents:0': image_data2})
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
    timeCheck() #17 second
    ##################################

if __name__ == "__main__":
    c = NScapture(video_show=True)
    c.run()
    c.stop()

