import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='read video')
parser.add_argument('--video_0', type=str, default='D:/work_lzm/everyBodyDance/data/result/jilejingtu_synthesized.avi')
parser.add_argument('--video_1', type=str,
                    default='D:/work_lzm/everyBodyDance/data/source_videos/source_jilejingtu.mp4')

parser.add_argument('--video_out', type=str, default='../data/out2.avi')

args = parser.parse_args()

cap_0 = cv2.VideoCapture(args.video_0)
cap_1 = cv2.VideoCapture(args.video_1)

frameSize_0 = (int(cap_0.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_0.get(cv2.CAP_PROP_FRAME_HEIGHT)))
frameSize_1 = (int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT)))

if cap_0.isOpened() is False or cap_1.isOpened() is False:
    print("Error opening video stream or file")
    exit(-1)

frameSize = (0, 0)
if (frameSize_0[1] < frameSize_1[1]):
    r = frameSize_0[1] / frameSize_1[1]
    w = int(round(frameSize_1[0] * r)) + frameSize_0[0]
    # h_out = int(round(w * 9 / 16))
    h_out = frameSize_0[1]
    frameSize = (w, h_out)
else:
    r = frameSize_1[0] / frameSize_0[1]
    w = int(round(frameSize_0[0] * r)) + frameSize_1[0]
    # h_out = int(round(w * 9 / 16))
    h_out = frameSize_1[1]
    frameSize = (w, h_out)

out = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc(*'XVID'), 25, frameSize)

while cap_0.isOpened() and cap_1.isOpened():
    b0, image_0 = cap_0.read()
    b1, image_1 = cap_1.read()

    if (b0 == False or b1 == False):
        break
    if (frameSize_0[1] < frameSize_1[1]):
        r = frameSize_0[1] / frameSize_1[1]
        image_1 = cv2.resize(image_1, None, fx=r, fy=r)
    else:
        r = frameSize_1[1] / frameSize_0[1]
        image_0 = cv2.resize(image_0, None, fx=r, fy=r)

    image = np.concatenate((image_1, image_0), axis=1)
    # image=np.pad(image,((270,270),(0,0),(0,0)),'constant')
    cv2.imshow('real image ', image)
    if cv2.waitKey(4) == 27:
        break
    out.write(image)
