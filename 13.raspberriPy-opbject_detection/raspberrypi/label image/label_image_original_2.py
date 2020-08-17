
import tensorflow as tf
import sys
# change this as you see fit
image_path_1 = sys.argv[1]
image_path_2 = sys.argv[2]
image_path_3 = sys.argv[3]
print("image_path: ", image_path_1 )
print("image_path: ", image_path_2 )
print("image_path: ", image_path_3 )
# Read in the imag흐gfile.FastGFile(image_path_1, 'rb').read()
image_data_2 = tf.gfile.FastGFile(image_path_2, 'rb').read()
image_data_3 = tf.gfile.FastGFile(image_path_3, 'rb').read()
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("/Users/do-hyungkwon/GoogleDrive/deeplink/5.DoHyung/raspberrypi/restoreFolder/output_labels.txt")]
# Unpersists graph from file
with tf.gfile.FastGFile("/Users/do-hyungkwon/GoogleDrive/deeplink/5.DoHyung/raspberrypi/restoreFolder/output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')


    predictions = sess.run(softmax_tensor, \
                           {'DecodeJpeg/contents:0': image_data_1})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
    print()



    predictions = sess.run(softmax_tensor, \
                           {'DecodeJpeg/contents:0': image_data_2})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
    print()



    predictions = sess.run(softmax_tensor, \
                           {'DecodeJpeg/contents:0': image_data_3})

    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))

