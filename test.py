# USAGE
# python test.py '{"images":[{"id": "example_01.jpg","url": "images/example_01.jpg"}]}'

# import the necessary packages
import numpy as np
import argparse
import cv2
import sys
import json

def main():
    # print(sys.argv[1])
    data=json.loads(sys.argv[1])
    confidence_threshold = 0.2
    prototxt = './MobileNetSSD_deploy.prototxt.txt'
    model = './MobileNetSSD_deploy.caffemodel'
    # initialize the list of class labels MobileNet SSD was trained to
    # detect, then generate a set of bounding box colors for each class
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load our serialized model from disk
    # print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation)
    out = {}
    out['images'] = []
    for image in data['images']:
        out_dict = {}
        out_dict['id'] = image['id']
        image_path = image['url']
        image = cv2.imread(image_path)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        # print("[INFO] computing object detections...")
        net.setInput(blob)
        detections = net.forward()
        out_dict['objects'] = []
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            obj_dict = {}
            obj_dict['id'] = i.item()
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > confidence_threshold:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                # label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                obj_dict['class'] = CLASSES[idx]
                obj_dict['coordinates'] = {
                                       "x":startX.item(),
                                       "y":startY.item(),
                                       "h":endY.item(),
                                       "w":endX.item()                     
                                       }
                out_dict['objects'].append(obj_dict)
                # print("[INFO] {}".format(label))
        out['images'].append(out_dict)
    # print(out)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
