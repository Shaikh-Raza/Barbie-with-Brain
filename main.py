import cv2
import numpy as np
from twilio.rest import Client
import time
import RPi.GPIO as GPIO
from time import sleep
thres = 0.45 # Threshold to detect object
nms_threshold = 0.2


# For IR Sensor or servo motor

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(3, GPIO.IN)
GPIO.setup(5, GPIO.IN)
GPIO.setup(40,GPIO.OUT)


cap = cv2.VideoCapture(0)

cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)
 


classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    # classNames = f.read().rstrip('n').split('n')
    classNames =[line.strip() for line in f]


configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



while True:
     
    ret,img = cap.read()
#     ret, frame = cap.read()

    ret = cv2.resize(img, (0,0), fx=0.5,fy=0.5)
#     cv2.imshow("Frame",frame)

    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    indices = cv2.dnn.NMSBoxes(bbox,confs,thres,nms_threshold)

    print(indices)

    for i in indices:
        # i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3] 
        cv2.rectangle(img, (x,y),(x+w,h+y), color=(0, 255, 0), 
        thickness=2) 
        cv2.putText(img,classNames[classIds[i]-1].upper(),(box[0]+10,box[1]+30), 
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.putText(img,classNames[i].upper(),(box[0]+10,box[1]+30),
        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        new_data = img,classNames[classIds[i]-1]
        SID = 'ACe5801c0d83ea98579f7c811d2d989965'
        AUTH_TOKEN = '224520fdf0d0a67acddcf6a4e17caa42'
        for name in new_data:
            print(name)
            if name == 'scissor':
                cl = Client(SID, AUTH_TOKEN) 
                cl.messages.create(body='Hey, Abdul Raza Scissor Detected!! ⚠ ☠ ⚠ "Look your child has scissored, put them away,it is dangerous for the child".', from_ ='+16074008278', to='+918070130325')
                
            elif name == 'knife':

                cl = Client(SID, AUTH_TOKEN) 
                cl.messages.create(body='Hey, Abdul Raza Knife Detected!! ⚠ ☠ ⚠ "Look your child has Knife, put them away,it is dangerous for the child".', from_ ='+16074008278', to='+918070130325')
#             elif name == 'cell phone':

#                 cl = Client(SID, AUTH_TOKEN) 
#                 cl.messages.create(body='Hey, Abdul Raza Cell Phone Detected!! ⚠ ☠ ⚠ "Look your child has Cell Phone, put them away,it is dangerous for the child".', from_ ='+16205319591', to='+919768107866')
    
    right = GPIO.input(3)
    left = GPIO.input(5)

    if right == 1:
#         p.ChangeDutyCycle(5)
#         time.sleep(0.5)
        GPIO.output(40, GPIO.HIGH)
        sleep(1)
        GPIO.output(40, GPIO.LOW)

    if left == 1:
        
        GPIO.output(40, GPIO.HIGH)
        sleep(1)
        GPIO.output(40, GPIO.LOW)
    
            
   # cv2.imshow("Output",ret)
    cv2.waitKey(1)
