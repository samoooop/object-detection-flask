from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import base64_to_image
import cv2
from DetectorAPI import DetectorAPI

threshold = 0.4

app = Flask(__name__)
CORS(app)

model_path = '/home/samoooop/workspace/flask/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)

@app.route("/hello", methods = ['POST'])
def test():
    print(request.form['image'])
    return "hello "

@app.route("/", methods=['GET', 'POST'])
def hello():
    # body = request.get_json()
    # print(body.get('image'))
    # print(request.form)
    # print(request.get_json(force = True))
    # print(request.args)
    base64_image = request.form.get('image')
    if base64_image is None:
        return 'IMAGE_IS_REQUIRED', 400
    img = base64_to_image(base64_image)
    rgb_img = img.copy()
    rgb_img = rgb_img[...,::-1]
    boxes, scores, classes, num = odapi.processFrame(rgb_img)
    colors = [
        (255,0,0),
        (0,255,0),
        (0,0,255),
        (1,255,255),
        (255,1,255),
        (255,255,1)
    ]
    mood_count = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(boxes)):
        # Class 1 represents human
        print(scores[i])
        if scores[i] > threshold:
            box = boxes[i]
            mood_count[classes[i]] += 1
            cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),colors[classes[i] - 1],2)
    cv2.imwrite("test.jpg", img)
    result = jsonify(
        common = mood_count[1],
        stress = mood_count[2],
        sad = mood_count[3],
        angry = mood_count[4],
        sleep = mood_count[5],
        happy = mood_count[6],
    )
    # cv2.imwrite("test.jpg", img)
    return result
