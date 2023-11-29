from flask import Flask, render_template, Response
import cv2
import numpy as np

classNames = {0: 'background',
              1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
              7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
              13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
              18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
              24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
              32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
              37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
              41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
              46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
              51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
              56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
              61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
              67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
              75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven',
              80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
              86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value
cap = cv2.VideoCapture(0)
model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb',
                                      'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # 동영상 파일 사용
def generate_frames():
    while True:
        success, frame = cap.read()  # 웹캠에서 프레임을 읽어옵니다.
        if not success:
            break
        else:
            frame = cv2.resize(frame,(640,360))
            img_height, img_width, _ = frame.shape
            model.setInput(cv2.dnn.blobFromImage(frame, size=(300,300),swapRB=True))
            output = model.forward()

            for detection in output[0, 0, :, :]:
                confidence = detection[2]
                if confidence > 0.5:
                    class_id = detection[1]
                    class_name = id_class_name(class_id, classNames)
                    box_x = detection[3] * img_width
                    box_y = detection[4] * img_height
                    box_width = detection[5] * img_width
                    box_height = detection[6] * img_height
                    cv2.rectangle(frame, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                    cv2.putText(frame, class_name, (int(box_x), int(box_y + 0.05 * img_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.005 * img_width, (0, 0, 255))
            cv2.imshow('dd',frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # 프레임을 바이트 스트림으로 전송


@app.route('/')
def index():
    return render_template('index.html')  # HTML 템플릿을 렌더링합니다.

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0')
