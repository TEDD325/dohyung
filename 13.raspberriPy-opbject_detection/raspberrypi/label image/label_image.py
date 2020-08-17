
import tensorflow as tf
import sys
import os
import time
import subprocess


# Loads label file, strips off carriage return
# label_lines = [line.rstrip() for line in tf.gfile.GFile("/home/pi/tf_files/output_labels.txt")]
label_lines = [line.rstrip() for line in tf.gfile.GFile("/Users/do-hyungkwon/GoogleDrive/github/deeplink/5.DoHyung/raspberrypi/restoreFolder/output_labels.txt")]
label_lines = [line.rstrip() for line in tf.gfile.GFile("/home/pi/tf_files/output_labels.txt")]
# Unpersists graph from file
with tf.gfile.FastGFile("/Users/do-hyungkwon/GoogleDrive/github/deeplink/5.DoHyung/raspberrypi/restoreFolder/output_graph.pb", 'rb') as f:
    try :
        graph_def = tf.GraphDef()
    except MemoryError as err:
        print("Memory Error Occured")
        subprocess.call(["/home/pi/Desktop/restart.sh"])

    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')


    spath = r"/Users/do-hyungkwon/GoogleDrive/github/deeplink/5.DoHyung/raspberrypi/file_detect_prac/"
    # spath = r"/home/pi/tf_files/test/"

    while True:
        for roots, dirs, files in os.walk(spath):
            idx = len(files)
            print("File = %s" % files[idx-1])
            image_path = spath + files[idx-1]
            image_data = tf.gfile.FastGFile(image_path, 'rb').read()

            predictions = sess.run(softmax_tensor, \
                                   {'DecodeJpeg/contents:0': image_data})

            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))

        time.sleep(3)






