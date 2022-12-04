import torch
import cv2
import numpy as np

from Tools.byte_tracker import BYTETracker

model = torch.hub.load('ultralytics/yolov5','custom', 'yolov5s.pt')
tracker = BYTETracker(0.5, 30, 0.5,frame_rate = 30)

cap = cv2.VideoCapture("sample.mp4")
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model(frame).xyxyn[0].to_numpy().cpu().numpy()
    result[:,:2] *= np.array([width,height])
    result[:,2:4] *= np.array([width,height])
    
    online_targets = tracker.update(result[:, :-1], [height, width], [height, width])
    online_tlwhs = []
    online_ids = []
    online_scores = []
    online_centroids = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        centroid = t.tlwh_to_xyah(tlwh)[:2]
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > 10 and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            online_centroids.append(centroid)
