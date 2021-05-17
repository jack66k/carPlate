import cv2 as cv
import numpy as np


class yolo:
    def __init__(self):
        # Initialize the parameters
        self.confThreshold = 0.5  # Confidence threshold
        self.nmsThreshold = 0.4  # Non-maximum suppression threshold

        self.inpWidth = 416  # 608     #Width of network's input image
        self.inpHeight = 416  # 608     #Height of network's input image

        self.modelConfiguration = 'yolov3-KD.cfg'
        self.modelWeights = 'yolov3-KD_last.weights'

        self.net = cv.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)

        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def getOutputsNames(self, net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 绘制车牌框
    def drawPred(self, left, top, right, bottom, frame):
        frame = cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        return frame

    # 返回车牌框
    def returnPred(self, frame, left, top, right, bottom):
        targ = frame[top:bottom, left:right]
        return targ

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        plate_list = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            frame = self.drawPred(left, top, left + width, top + height, frame)
            plate_list.append(self.returnPred(frame, left, top, left + width, top + height))
        return frame, plate_list

    def return_frame(self, frame):
        """
        """
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.getOutputsNames(self.net))
        return self.postprocess(frame, outs)
