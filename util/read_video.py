import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='read video')
parser.add_argument('--video', type=str, default='data/3.mp4')
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)

if cap.isOpened() is False:
    print("Error opening video stream or file")
step = 0
done = False
total_steps=0
while cap.isOpened() and done == False:
    ret_val, image = cap.read()
    total_steps+=1

    cv2.imshow('real image ', image)

    if cv2.waitKey(50) == 27:
        break
    # number_str = format(total_steps,'04d')
    # cv2.imwrite('./images/' + str(total_steps) + "i.png", image)

    tmp_step = step
    while tmp_step > 0:
        tmp_step -= 1
        # cap.read()
        f = cap.grab()
        if f == False:
            done = True
            break