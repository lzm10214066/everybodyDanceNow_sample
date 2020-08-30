import json
import cv2
import numpy as np
import os
import argparse
import math
import collections
from collections import deque

keyPoints = collections.namedtuple("KeyPoints",
                                   ["pose_points", "face_points", "hand_left_points", "hand_right_points", "valid"])


def showPoints(image, points, color=(255, 255, 255), delay=0):
    if image is None:
        return
    for i, p in enumerate(points):
        x = int(points[i][0])
        y = int(points[i][1])
        cv2.circle(image, (x, y), 8, color, -1)
        # cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
    if delay != -1:
        cv2.imshow("test", image)
        cv2.waitKey(delay)


def get_points(points_file):
    with open(points_file, 'r') as load_f:
        points_data = json.load(load_f)

    if len(points_data['people']) < 1:
        return keyPoints(pose_points=-1, face_points=-1, hand_left_points=-1, hand_right_points=-1, valid=-1)
    pose_points = np.array(points_data['people'][0]['pose_keypoints_2d']).reshape([-1, 3])
    face_points = np.array(points_data['people'][0]['face_keypoints_2d']).reshape([-1, 3])
    hand_left_points = np.array(points_data['people'][0]['hand_left_keypoints_2d']).reshape([-1, 3])
    hand_right_points = np.array(points_data['people'][0]['hand_right_keypoints_2d']).reshape([-1, 3])

    # face_points_left=[8,68,69,27,28,29,30,48,54,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    face_points_left = [5, 6, 7, 8, 9, 10, 11, 68, 69, 0, 16, 27, 28, 29, 30]
    for i in range(70):
        if i not in face_points_left:
            face_points[i] = 0

    if len(pose_points) != 25 or len(hand_left_points) != 21 or len(hand_right_points) != 21 or len(face_points) != 70:
        print("pose_points:", len(pose_points),
              "face_points:", len(face_points),
              "hand_left_points:", len(hand_left_points),
              "hand_right_points:", len(hand_right_points))
        exit(-1)

    return keyPoints(pose_points=pose_points, face_points=face_points, hand_left_points=hand_left_points,
                     hand_right_points=hand_right_points, valid=1)


def make_dataset(dir):
    res = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            res.append(path)
    return res


def getMedianPoints(windows_points):
    pose_points_list = []
    face_points_list = []
    hand_left_points_list = []
    hand_right_points_list = []
    for i in range(len(windows_points)):
        pose_points_list.append(windows_points[i].pose_points)
        face_points_list.append(windows_points[i].face_points)
        hand_left_points_list.append(windows_points[i].hand_left_points)
        hand_right_points_list.append(windows_points[i].hand_right_points)

    pose_points = np.array(pose_points_list)
    pose_points_median = np.median(pose_points, axis=0)

    face_points = np.array(face_points_list)
    face_points_median = np.median(face_points, axis=0)

    hand_left_points = np.array(hand_left_points_list)
    hand_left_points_median = np.median(hand_left_points, axis=0)

    hand_right_points = np.array(hand_right_points_list)
    hand_right_points_median = np.median(hand_right_points, axis=0)

    # if pose_points_median[4][0] > 0 and pose_points_median[4][1] > 0:
    #     hand_right_points_median[0] = pose_points_median[4]
    #     hand_right_points_median[0] = pose_points_median[4]
    #
    # if pose_points_median[7][0] > 0 and pose_points_median[7][1] > 0:
    #     hand_left_points_median[0] = pose_points_median[7]
    #     hand_left_points_median[0] = pose_points_median[7]

    return keyPoints(pose_points=pose_points_median, face_points=face_points_median,
                     hand_left_points=hand_left_points_median, hand_right_points=hand_right_points_median, valid=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='crop body')
    parser.add_argument('--indir', type=str, default='../data/datasets/lzm')
    parser.add_argument('--outdir', type=str, default='../data/datasets/lzm_smoothed')
    parser.add_argument('--win_size', type=int, default=7)

    args = parser.parse_args()

    keypoints_jsons_paths = sorted(make_dataset(args.indir))
    windows = deque()
    if (os.path.exists(args.outdir) == False):
        os.mkdir(args.outdir)

    for p in keypoints_jsons_paths:
        print(p)
        # black=np.zeros((1024,1920,3),dtype=np.uint8)
        black = None
        points = get_points(p)
        showPoints(black, points.hand_right_points, (0, 0, 255))

        if points.valid == -1:
            continue
        windows.append(points)
        if len(windows) > args.win_size:
            windows.popleft()
        if len(windows) == args.win_size or True:
            points_smoothed = getMedianPoints(windows)
            pose_str = ''
            for i in range(25):
                for j in range(3):
                    pose_str += str(points_smoothed.pose_points[i][j]) + ','
            pose_str = pose_str.rstrip(',')

            face_str = ''
            for i in range(70):
                for j in range(3):
                    face_str += str(points_smoothed.face_points[i][j]) + ','
            face_str = face_str.rstrip(',')

            hand_left_str = ''
            for i in range(21):
                for j in range(3):
                    hand_left_str += str(points_smoothed.hand_left_points[i][j]) + ','
            hand_left_str = hand_left_str.rstrip(',')

            hand_right_str = ''
            for i in range(21):
                for j in range(3):
                    hand_right_str += str(points_smoothed.hand_right_points[i][j]) + ','
            hand_right_str = hand_right_str.rstrip(',')

            p_name = os.path.basename(p)
            out_path = os.path.join(args.outdir, p_name.split('.')[0] + ".txt")
            with open(out_path, 'w') as fout:
                fout.write(pose_str + '\n')
                fout.write(face_str + '\n')
                fout.write(hand_left_str + '\n')
                fout.write(hand_right_str)
