import cv2
import numpy as np
# Pretrained classes in the model
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

def DetectLineSlope(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    can = cv2.Canny(gray, 50, 200, None, 3)
    height = can.shape[0]
    rectangle = np.array([[(0, height), (120, 280), (520, 280), (640, height)]]) #[왼쪽 위, 왼쪽 아래, 오른쪽 위, 오른쪽 아래]
    mask = np.zeros_like(can)
    cv2.fillPoly(mask, rectangle, 255)
    masked_image = cv2.bitwise_and(can, mask)
    ccan = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)

    line_arr = cv2.HoughLinesP(masked_image, 1, np.pi / 180, 20, minLineLength=10, maxLineGap=10)
    line_R = np.empty((0, 5), int)  
    line_L = np.empty((0, 5), int)  
    if line_arr is not None:
        line_arr2 = np.empty((len(line_arr), 5), int)
        for i in range(0, len(line_arr)):
            temp = 0
            l = line_arr[i][0]
            line_arr2[i] = np.append(line_arr[i], np.array((np.arctan2(l[1] - l[3], l[0] - l[2]) * 180) / np.pi))
            if line_arr2[i][1] > line_arr2[i][3]:
                temp = line_arr2[i][0], line_arr2[i][1]
                line_arr2[i][0], line_arr2[i][1] = line_arr2[i][2], line_arr2[i][3]
                line_arr2[i][2], line_arr2[i][3] = temp
            if line_arr2[i][0] < 320 and (abs(line_arr2[i][4]) < 170 and abs(line_arr2[i][4]) > 95):
                line_L = np.append(line_L, line_arr2[i])
            elif line_arr2[i][0] > 320 and (abs(line_arr2[i][4]) < 170 and abs(line_arr2[i][4]) > 95):
                line_R = np.append(line_R, line_arr2[i])
    line_L = line_L.reshape(int(len(line_L) / 5), 5)
    line_R = line_R.reshape(int(len(line_R) / 5), 5)

    try:
        line_L = line_L[line_L[:, 0].argsort()[-1]]
        degree_L = line_L[4]
        cv2.line(ccan, (line_L[0], line_L[1]), (line_L[2], line_L[3]), (255, 0, 0), 10, cv2.LINE_AA)
    except:
        degree_L = 0
    try:
        line_R = line_R[line_R[:, 0].argsort()[0]]
        degree_R = line_R[4]
        cv2.line(ccan, (line_R[0], line_R[1]), (line_R[2], line_R[3]), (255, 0, 0), 10, cv2.LINE_AA)
    except:
        degree_R = 0
    mimg = cv2.addWeighted(src, 1, ccan, 1, 0)
    return mimg, degree_L, degree_R


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value
cap = cv2.VideoCapture('ccc.mp4')
model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb',
                                      'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame,(640,360))
        img_height, img_width, _ = frame.shape
        model.setInput(cv2.dnn.blobFromImage(frame, size=(300,300),swapRB=True))
        output = model.forward()

        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > 0.5:
                class_id = detection[1]
                class_name = id_class_name(class_id, classNames)
                print(str(class_id) + " " + str(confidence) + " " + class_name)
                box_x = detection[3] * img_width
                box_y = detection[4] * img_height
                box_width = detection[5] * img_width
                box_height = detection[6] * img_height
                cv2.rectangle(frame, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                cv2.putText(frame, class_name, (int(box_x), int(box_y + 0.05 * img_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.005 * img_width, (0, 0, 255))
        cv2.imshow('dd',frame)
        cv2.imshow('dd',DetectLineSlope(frame)[0])
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

