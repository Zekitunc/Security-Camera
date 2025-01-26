# python yolo.py --image resim/arabalar.jpg --yolo main

import numpy as np
import argparse
import time
import cv2
import os

# yapını çağrılması için gerekli olan bilgiler
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# sınıf isimleri
isimler = os.path.sep.join([args["yolo"], "isimler.names"])
LABELS = open(isimler).read().strip().split("\n")

# renk ayarları
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# yolo Last
weights = os.path.sep.join([args["yolo"], "yoloSon.weights"])
config = os.path.sep.join([args["yolo"], "yoloSon.cfg"])


# alternatif yoloV4-TİNY
#weights = os.path.sep.join([args["yolo"], "yolov4-tiny.weights"])
#config
# = os.path.sep.join([args["yolo"], "yolov4-tiny.cfg"])


#alternatif yoloV3
#weights = os.path.sep.join([args["yolo"], "yolov3.weights"])
#config
# = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# yolonun diskten yüklenmesi
print("[INFO] YOLO yükleniyor...")
net = cv2.dnn.readNetFromDarknet(config, weights)

image = cv2.imread(args["image"])
(H, W) = image.shape[:2]

ln = net.getLayerNames() 
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

# aldığı süre
print("YOLO {:.6f} saniye sürdü.".format(end - start))


boxes = []
confidences = []
classIDs = []

#döngünün katmansal olarak nesneden ne kadar emin olduğu ve sınıfı gibi belli parametrelerin
#belirlendiği asıl bölge
for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		if confidence > args["confidence"]:

			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")

			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# üstüste kutuları thresholdladık
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# en az bir görüntü var
if len(idxs) > 0:
	for i in idxs.flatten():
		# kutunun pozisyonunu belirler
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# resmi al ve işaretle
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# çıktı ver
cv2.imshow("Image", image)
cv2.waitKey(0)