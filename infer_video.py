import os
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from utils import label_map_util
from utils import visualization_utils as vis_util


def load_model(model_path, labels_path, n_classes=90):
    # load model:
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # load labels:
    label_map = label_map_util.load_labelmap(labels_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=n_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return detection_graph, category_index

def detect_video(detection_graph, category_index, vid):
    with detection_graph.as_default():
        with tf.Session( graph=detection_graph ) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name( 'image_tensor:0' )
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name( 'detection_boxes:0' )
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name( 'detection_scores:0' )
            detection_classes = detection_graph.get_tensor_by_name( 'detection_classes:0' )
            num_detections = detection_graph.get_tensor_by_name( 'num_detections:0' )
            while True:
                ok, image = vid.read()
                if not ok:
                    break
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims( image, axis=0 )
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded} )
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze( boxes ),
                    np.squeeze( classes ).astype( np.int32 ),
                    np.squeeze( scores ),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8 )
                cv2.imshow( 'image', image)
                cv2.waitKey(1)



def main():
    MODEL_NAME = 'frcnn_resnet50_anatomy'
    model_path = 'models/' + MODEL_NAME + '/frozen_inference_graph.pb'
    labels_path = os.path.join( 'data/labels_mapping', 'anatomy_label_map.pbtxt' )
    n_classes = 1

    sample_video = cv2.VideoCapture('/media/dotan/Data/theator/datasets/theator-lapchole/MLP/MPL_002/MPL_LC(091217)_002.mp4')
    detection_graph, category_index = load_model(model_path, labels_path, n_classes)
    detect_video(detection_graph, category_index, sample_video)


if __name__ == '__main__':
    main()