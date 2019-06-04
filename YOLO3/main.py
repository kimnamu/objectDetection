import argparse
from yolo import YOLO, detect_video, detect_live
from PIL import Image
from io import BytesIO

import tensorflow as tf


global graph
graph = tf.get_default_graph()

FLAGS = None
# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

## Command line options
parser.add_argument(
    '--model', type=str,
    help='path to model weight file, default ' + YOLO.get_defaults("model_path")
)

parser.add_argument(
    '--anchors', type=str,
    help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
)

parser.add_argument(
    '--classes', type=str,
    help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
)

parser.add_argument(
    '--gpu_num', type=int,
    help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
)

parser.add_argument(
    '--image', default=False, action="store_true",
    help='Image detection mode, will ignore all positional arguments'
)

# Command line positional arguments -- for video detection mode

parser.add_argument(
    "--input", nargs='?', type=str, required=False, default='./path2your_video',
    help="Video input path"
)

parser.add_argument(
    "--output", nargs='?', type=str, default="",
    help="[Optional] Video output path"
)

FLAGS = parser.parse_args()
yolo = YOLO(**vars(FLAGS))

def objectDetection(filename):
    image = Image.open(filename)

    with graph.as_default():
        r_image, result = yolo.detect_image(image)

    final_result = []
    for r in result:
        if r[1] > 0.5:
            Flag = True
            for dupl in final_result:
                if dupl == r[0]:
                    Flag = False
                    break
            if Flag:
                final_result.append(r[0])

    return {"object": final_result}

print(objectDetection('./test.jpeg'))