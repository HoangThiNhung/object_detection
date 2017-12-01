
# coding: utf-8

# In[1]:


import numpy as np
import argparse
import cv2
 


# In[2]:


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


# In[3]:


net = cv2.dnn.readNetFromCaffe("data/models/MobileNetSSD_deploy.prototxt.txt", "data/models/MobileNetSSD_deploy.caffemodel")


# In[4]:


image = cv2.imread("data/test/test.jpg")
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
    (300, 300), 127.5)


# In[5]:


net.setInput(blob)
detections = net.forward()


# In[ ]:


for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
 
    idx = int(detections[0, 0, i, 1])
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")

    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
    print("[INFO] {}".format(label))
    cv2.rectangle(image, (startX, startY), (endX, endY),
    COLORS[idx], 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(image, label, (startX, y),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


# In[ ]:


cv2.imshow("Output", image)
cv2.waitKey(0)

