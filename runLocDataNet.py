import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd
import torchvision
import locDataNet
import math
import yolov4
import sys
import os

def initNetwork():
    model = locDataNet.LocDataNet(4, 2)
    weight = './Weights/locDataNet.pth'
    model.load_state_dict(torch.load(weight))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def count_frames(vid_path):
    cap = cv2.VideoCapture(vid_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, length, fps

def passFramesToNet(cap, frame_div):
    count = 0
    df = pd.DataFrame(columns=['x', 'y', 'w', 'h'])

    yolo, class_names = yolov4.initYOLO()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            try:
                box = yolov4.getBoundingBoxData(yolo, frame, class_names)
                df1 = pd.DataFrame({'x':[box[0]], 'y':[box[1]], 'w':[box[2]], 'h':[box[3]]})

                df = df.append(df1, ignore_index=True)
            except:
                height, width = frame.shape[:2]
                df1 = pd.DataFrame({'x':[width/2], 'y':[height/2], 'w':[5], 'h':[5]})
                df = df.append(df1, ignore_index=True)

            count += frame_div
            cap.set(1, count)

        else:
            cap.release()
            break

    return df


def calcCoordinates(d, theta, phi, lambd):
    R = 6371e3
    theta = (theta*180/math.pi + 360)%360
    theta = theta * math.pi / 180
    d = d/1000
    phi = phi*math.pi/180
    lambd = lambd*math.pi/180
    phi2 = math.asin(math.sin(phi)*math.cos(d/R) + math.cos(phi)*math.sin(d/R)*math.cos(theta))
    lambda2 = lambd + math.atan2(math.sin(theta)*math.sin(d/R)*math.cos(phi), math.cos(d/R)-math.sin(phi)*math.sin(phi2))

    phi2 = phi2 * 180 / math.pi
    lambda2 = lambda2 * 180 / math.pi

    return phi2, lambda2

def runModel(model, inputs, phi, lambd, timeIncr, path=None):
    out = pd.DataFrame(columns=['time', 'lat', 'lon'])
    time = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inVal = torch.from_numpy(inputs.to_numpy()).float()
    inVal = inVal.view(-1, 4)
    model.to(device)
    model = model.float()
    inVal.to(device)

    if torch.cuda.is_available():
        outputs = model(inVal.cuda())
    else:
        outputs = model(inVal)
    valOutputs = outputs.cpu().detach().numpy()
    for coords in valOutputs:       
        d = coords[0]
        theta = coords[1]
        phi2, lambda2 = calcCoordinates(d, theta, float(phi), float(lambd))
        out = out.append({'time':round(time,1), 'lat':round(phi2, 9), 'lon':round(lambda2, 9)}, ignore_index=True)
        time += timeIncr

    if path == None:
        out.to_csv('./output.csv', header=False, index=False)
    else:
        if os.path.exists(path):
            out.to_csv( path + 'output.csv', header=False, index=False)
        else:
            print('Specified output directory does not exist')

    

if __name__ == '__main__':
    if len(sys.argv) >= 5:
        cap, frameNo, fps = count_frames(sys.argv[1])
        frame_div = frameNo / int(sys.argv[2])

        length = frameNo / fps
        timeIncr = length / int(sys.argv[2])

        data = passFramesToNet(cap, math.floor(frame_div))
        model = initNetwork()

        if len(sys.argv) == 5:
            runModel(model, data, sys.argv[3], sys.argv[4], timeIncr)
        elif len(sys.argv) == 6:
            runModel(model, data, sys.argv[3], sys.argv[4], timeIncr, sys.argv[5])
        else:
            print('Wrong number of input arguments')

        print('Output successfully compiled to output.csv file')

    else:
        print("Wrong number of argumemts: 'python3 runLocDataNet.py <path_to_vid_file> <no. of points> <lat of camera> <lon of camera>'")