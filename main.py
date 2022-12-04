# Import necessary packages
import time

import cv2
import csv
import numpy as np
from tracker import *
from datetime import datetime, timedelta
import CVCascadeFilter
from threading import Timer


# Initialize Tracker
tracker = EuclideanDistTracker()
greenack = 0
redack = 1
redack1 = 1
greenack1 = 0
dt2 = 0
dt1 = 0
# Initialize the videocapture object
cap = cv2.VideoCapture('ind.mp4')
input_size = 320

# Detection confidence threshold
confThreshold = 0.8
nmsThreshold = 0.5

font_color = (52,235,198)
font_size = 0.5
font_thickness = 2

# Middle cross line position
middle_line_position = 300
up_line_position = middle_line_position - 10
down_line_position = middle_line_position + 10

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')

# class index for our required detection classes
required_class_index = [2, 3, 5, 7]

detected_classNames = []

# Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

# Configure the network backenda

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')


def redLight(img):
    global redack
    global redack1
    global greenack
    global greenack1
    global dt2
    global dt1

    if redack == 1:
        if redack1 == 1:
            dt2 = int(("%s" % datetime.now().second))
            redack1 = 0
        dt1 = int(("%s" % datetime.now().second))
        #print("dt1",dt1,dt2)
        # print("dt2",dt2)
        cc = 0
        cc2 = 0
        if dt1 > dt2:
            cc = dt1
            cc2 = dt2
        else:
            cc = dt2
            cc2 = dt1
        if cc < (cc2 + 20):
            cv2.circle(img, (530, 30), 14, (0, 0, 255), -1)
            #cv2.circle(img, (430, 30), 14, (0, 0, 255), -1)
            cv2.circle(img, (430, 65), 14, (0, 255, 0), -1)
            str1 = str(dt1 - dt2)
            cv2.putText(img, str1, (550, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            cv2.putText(img, str1, (450, 69), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            greenack = 0
        else:
            greenack = 1
            greenack1 = 1
            redack = 0


def greenLight(img, density):
    global redack
    global redack1
    global greenack
    global greenack1
    global dt2
    global dt1
    yu = 0
    if density < 10:
        yu = 5
    else:
        yu = 10
    if greenack == 1:
        if greenack1 == 1:
            dt2 = int(("%s" % datetime.now().second))
            greenack1 = 0
        dt1 = int(("%s" % datetime.now().second))
        #print("dt1",dt1, dt2)
        # print("dt2",dt2)
        cc = 0
        cc2 = 0
        if dt1 > dt2:
            cc = dt1
            cc2 = dt2
        else:
            cc = dt2
            cc2 = dt1
        if cc < cc2 + yu:
            cv2.circle(img, (530, 65), 14, (0, 255, 0), -1)
            #cv2.circle(img, (430, 65), 14, (0, 255, 0), -1)
            cv2.circle(img, (430, 30), 14, (0, 0, 255), -1)
            str1 = str(dt1 - dt2)
            cv2.putText(img, str1, (550, 69), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            cv2.putText(img, str1, (450, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, 1)
            redack = 0
        else:
            redack = 1
            redack1 = 1
            greenack = 0


# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# List for store vehicle count information
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0, 0]
down_list = [0, 0, 0, 0]
startTracker = {}  # STORE STARTING TIME OF CARS
endTracker = {}  # STORE ENDING TIME OF CARS
markGap = 20
fpsFactor = 3


# Function for count vehicle
def count_vehicle(density, box_id, img):
    x, y, w, h, id, index = box_id

    # Find the center of the rectangle for detection
    center = find_center(x, y, w, h)
    ix, iy = center
    # Find the current position of the vehicle
    if (iy > up_line_position) and (iy < middle_line_position):
        if id not in temp_up_list:
            startTracker[id] = datetime.now()
            temp_up_list.append(id)

    elif (iy < down_line_position) and (iy > middle_line_position):
        if id not in temp_down_list:
            temp_down_list.append(id)

    elif iy < up_line_position:
        if id in temp_down_list:
            temp_down_list.remove(id)
            up_list[index] = up_list[index] + 1

    elif iy > down_line_position:
        if id in temp_up_list:
            temp_up_list.remove(id)
            endTracker[id] = datetime.now()
            down_list[index] = down_list[index] + 1

    # Draw circle in the middle of the rectangle
    cv2.circle(img, center, 2, (0, 0, 255), -1)  # end here
    # print(up_list, down_list)


# Function for calculating the speed of vehicle
def estimateSpeed(position):
    timeDiff = int(CVCascadeFilter.CVCascadeFilter.timings(bin(position).replace("0b", "")), 2)
    # print("timeDiff", timeDiff)
    speed = round(markGap / timeDiff * fpsFactor * 3.6, 2)
    return speed


# Function for finding the detected objects from the network output
def postProcess(outputs, img):
    global detected_classNames
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)

                    w, h = int(det[2] * width), int(det[3] * height)
                    x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
                    boxes.append([x, y, w, h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    for i in indices.flatten():
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        # print(x,y,w,h)

        color = [int(c) for c in colors[classIds[i]]]
        name = classNames[classIds[i]]
        detected_classNames.append(name)
        # Draw classname and confidence score
        speed = estimateSpeed(i)
        cv2.putText(img, f'{name.upper()} {int(confidence_scores[i] * 100)}% {speed} Kmph',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw bounding rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        detection.append([x, y, w, h, required_class_index.index(classIds[i])])

    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    density = len(boxes_ids)
    for box_id in boxes_ids:
        count_vehicle(density, box_id, img)

    return density


def realTime():
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)

        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
        #outputNames = net.getUnconnectedOutLayers()
        # Feed data to the network
        outputs = net.forward(outputNames)

        # Find the objects from the network output
        den = postProcess(outputs, img)
        # print("denstity", den)
        # Draw the crossing lines
        cv2.line(img, (500, 10), (560, 10), (0, 255, 0), 2)
        cv2.line(img, (500, 10), (500, 120), (0, 255, 0), 2)
        cv2.line(img, (560, 10), (560, 120), (0, 255, 0), 2)
        cv2.line(img, (500, 120), (560, 120), (0, 255, 0), 2)
        cv2.circle(img, (530, 30), 15, (0, 0, 0), 2)
        cv2.circle(img, (530, 65), 15, (0, 0, 0), 2)
        cv2.circle(img, (530, 100), 15, (0, 0, 0), 2)
        cv2.line(img, (400, 10), (460, 10), (0, 255, 0), 2)
        cv2.line(img, (400, 10), (400, 120), (0, 255, 0), 2)
        cv2.line(img, (460, 10), (460, 120), (0, 255, 0), 2)
        cv2.line(img, (400, 120), (460, 120), (0, 255, 0), 2)
        cv2.circle(img, (430, 30), 15, (0, 0, 0), 2)
        cv2.circle(img, (430, 65), 15, (0, 0, 0), 2)
        cv2.circle(img, (430, 100), 15, (0, 0, 0), 2)

        redLight(img)
        density = 0
        if 0 <= den <= 20:
            density = 5
        elif 20 <= den <= 40:
            density = 10
        elif (41 <= den):
            density = 15
        greenLight(img, density)
        cv2.line(img, (0, middle_line_position), (iw, middle_line_position), (255, 0, 255), 2)
        cv2.putText(img, "Up        ", (40, 270), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.line(img, (0, up_line_position), (iw, up_line_position), (28, 44, 253), 2)
        cv2.putText(img, "Down        ", (40, 340), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.line(img, (0, down_line_position), (iw, down_line_position), (23, 2, 7), 2)

        # Draw counting texts in the frame
        cv2.putText(img, "Up", (110, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Down", (160, 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Car:        " + str(up_list[0]) + "     " + str(down_list[0]), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Motorbike:  " + str(up_list[1]) + "     " + str(down_list[1]), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Bus:        " + str(up_list[2]) + "     " + str(down_list[2]), (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)
        cv2.putText(img, "Truck:      " + str(up_list[3]) + "     " + str(down_list[3]), (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Show the frames
        cv2.imshow('Output', img)

        if cv2.waitKey(1) == ord('q'):
            break

    # Write the vehicle counting information in a file and save it

    with open("data.csv", 'w') as f1:
        cwriter = csv.writer(f1)
        cwriter.writerow(['Direction', 'car', 'motorbike', 'bus', 'truck'])
        up_list.insert(0, "Up")
        down_list.insert(0, "Down")
        cwriter.writerow(up_list)
        cwriter.writerow(down_list)
    f1.close()
    # print("Data saved at 'data.csv'")
    # Finally realese the capture object and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    realTime()
    postProcess()
    count_vehicle()
