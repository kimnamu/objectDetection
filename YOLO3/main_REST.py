import argparse
from yolo import YOLO, detect_video, detect_live
from PIL import Image
from io import BytesIO

import tensorflow as tf

import requests

from flask import Flask, request
from flask_restful import Resource, Api
from flask_restful import reqparse
from werkzeug import secure_filename
import cv2
app = Flask(__name__)
api = Api(app)

def url_to_image(url):
    resp = requests.get(url)
    image = Image.open(BytesIO(resp.content))

    return image


def detect_img(yolo, url):
    image = url_to_image(url)
    r_image, result = yolo.detect_image(image)
    return result

def detect_img2(yolo, image):
    r_image, result = yolo.detect_image(image)
    return result

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

class objectDetectionURL(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('url', type=str)
        args = parser.parse_args()

        with graph.as_default():
            result = detect_img(yolo, args['url'])
        print(result)

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


class objectDetectionFile(Resource):
    def post(self):
        file = request.files['file']
        file.save(secure_filename(file.filename))
        image = Image.open(file.filename)

        with graph.as_default():
            result = detect_img2(yolo, image)

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

api.add_resource(objectDetectionURL, '/url')
api.add_resource(objectDetectionFile, '/file')

if __name__ == '__main__':
    app.run(debug=False)
