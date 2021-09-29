import cv2
import numpy as np

def initYOLO():
    class_names = []
    with open("./obj.names", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]


    #net = cv2.dnn_DetectionModel('yolov4.cfg', 'yolov4.weights')

    net = cv2.dnn.readNetFromDarknet('./yolo-obj.cfg', './Weights/yolo-obj_2000.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255)

    return model, class_names


def getBoundingBoxData(model, img, class_names):
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    flag = 0

    frame = img
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for (classid, score, box) in zip(classes, scores, boxes):
        if class_names[classid[0]] == 'boat':

            box = box.astype(np.float64)
            flag = 1
            return box

        else:
            flag = 0

    if flag == 0:
        return None