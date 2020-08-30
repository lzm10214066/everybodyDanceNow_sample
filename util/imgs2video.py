import cv2
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='read video')
parser.add_argument('--images_dir', type=str, default='../data/datasets/jixie2/test_stick')
parser.add_argument('--out_video_path', type=str, default='../data/datasets/jixie2/test.avi')
parser.add_argument('--scale', type=float, default=1.)
parser.add_argument('--frameRate', type=float, default=25)

args = parser.parse_args()

img_names = sorted(os.listdir(args.images_dir))
img0 = cv2.imread(os.path.join(args.images_dir, img_names[0]))
if args.scale != 1.0:
    img0 = cv2.resize(img0, None, None, fx=args.scale, fy=args.scale)
frameSize = (img0.shape[1], img0.shape[0])

out_video_path = args.out_video_path
out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'XVID'), args.frameRate, frameSize)

for n in img_names:
    p = os.path.join(args.images_dir, n)
    print('processing ', p)
    image = cv2.imread(p)
    if args.scale != 1.0:
        image = cv2.resize(image, None, None, fx=args.scale, fy=args.scale)
    # cv2.imshow('real image ', image)
    # if cv2.waitKey(10) == 27:
    #     break
    out.write(image)
