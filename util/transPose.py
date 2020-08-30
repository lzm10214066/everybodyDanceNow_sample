import cv2
import argparse
import numpy as np
import os
from collections import deque
import sys
sys.path.append("/data/bd-recommend/lizhenmao/everyBodyDance_work/everyBodyDance")
from util.smooth_points import get_points,getMedianPoints,make_dataset,keyPoints,showPoints

parser = argparse.ArgumentParser(description='trans points')
parser.add_argument('--source_points_dir', type=str, default='D:/download/source_jilejingtu_jsons')
parser.add_argument('--source_pose_norm', type=str, default='D:/download/source_pose_norm_jilejingtu_keypoints.json')
parser.add_argument('--target_pose_norm', type=str, default='D:/download/target_pose_norm_lzm_keypoints.json')
parser.add_argument('--out_dir', type=str, default='../data/transformed_jilejingtu_points')
parser.add_argument('--out_video', type=str, default='../data/trans.avi')
parser.add_argument('--frameSize', type=str, default='1024x1920')
parser.add_argument('--win_size', type=int, default=7)
parser.add_argument('--refer_image', type=str, default='D:/download/target_pose_norm_lzm.png')

args = parser.parse_args()

def get_body_box(points_xy):

    min_x = 1000000
    max_x = 0
    min_y = 1000000
    max_y = 0
    n_points=points_xy.shape[0]
    for p in range(n_points):
        x = points_xy[p][0]
        y = points_xy[p][1]
        if (x > 0 and y > 0):
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    return min_x, max_x, min_y, max_y

def getSimilarityTransform_fast(src, dst):
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    if (src.shape[0] != dst.shape[0]):
        print("Error: vectors not the same size")
        return warp_mat

    num_pairs = src.shape[0]
    print("finding similarity based on %d pairs of points" % num_pairs)

    min_x_s,max_x_s,min_y_s,max_y_s=get_body_box(src)
    min_x_d, max_x_d,min_y_d, max_y_d = get_body_box(dst)

    scale=(max_y_d-min_y_d)/(max_y_s-min_y_s)
    shift=((max_x_d+min_x_d)/2-(max_x_s+min_x_s)/2*scale,max_y_d-max_y_s*scale)

    warp_mat[0, 0] = scale
    warp_mat[0, 1] = 0
    warp_mat[0, 2] = shift[0]
    warp_mat[1, 0] = 0
    warp_mat[1, 1] = scale
    warp_mat[1, 2] = shift[1]

    return warp_mat

def getSimilarityTransform(src, dst):
    warp_mat = np.zeros((2, 3), dtype=np.float32)
    if (src.shape[0] != dst.shape[0]):
        print("Error: vectors not the same size")
        return warp_mat

    num_pairs = src.shape[0]
    print("finding similarity based on %d pairs of points" % num_pairs)

    srcMat = np.zeros((4, 2 * num_pairs), dtype=np.float32)
    dstMat = np.zeros((1, 2 * num_pairs), dtype=np.float32)

    for i in range(num_pairs):
        srcMat[0, i] = src[i, 0]
        srcMat[1, i] = -src[i, 1]
        srcMat[2, i] = 1
        srcMat[3, i] = 0

        srcMat[0, i + num_pairs] = src[i, 1]
        srcMat[1, i + num_pairs] = src[i, 0]
        srcMat[2, i + num_pairs] = 0
        srcMat[3, i + num_pairs] = 1

        dstMat[0, i] = dst[i][0]
        dstMat[0, i + num_pairs] = dst[i][1]

    # transform * src = dst
    # transform = dst * src' * inv(src * src')

    # A = dst * src'
    A = np.dot(dstMat, srcMat.T)

    # B = src * src'
    B = np.dot(srcMat, srcMat.T)

    # invert B...
    B_inv = np.matrix(B).I

    # multiply A and B^(-1)
    T = np.dot(A, B_inv)

    # set warp_mat
    T[0, 1]=0 #去掉旋转
    warp_mat[0, 0] = T[0, 0]
    warp_mat[0, 1] = -T[0, 1]
    warp_mat[0, 2] = T[0, 2]
    warp_mat[1, 0] = T[0, 1]
    warp_mat[1, 1] = T[0, 0]
    warp_mat[1, 2] = T[0, 3]

    return warp_mat

def transPoints(points,warp_mat):
    ones = np.ones((1, points.shape[0]))
    tmp_xy = np.concatenate((points[:, 0:2].T, ones), axis=0)
    tmp = np.dot(warp_mat, tmp_xy).T
    res = np.concatenate((tmp, points[:, 2:3]), axis=1)
    return res

if __name__ == '__main__':
    #get source pose and target pose
    points_left=[0,15,16,17,18,2,5,12,13,14,9,10,11,1]
    refer_image=cv2.imread(args.refer_image)
    source_pose_path=args.source_pose_norm
    target_pose_path=args.target_pose_norm

    black_s=np.zeros((1024,1920,3),dtype=np.uint8)
    source_points = get_points(source_pose_path)
    showPoints(black_s,source_points.pose_points,(255,255,0),-1)

    black_t=np.zeros((1024,1920,3),dtype=np.uint8)
    target_points = get_points(target_pose_path)
    showPoints(black_s, target_points.pose_points,(0,255,0),-1)

    source_pose=source_points.pose_points[points_left]
    target_pose=target_points.pose_points[points_left]

    black_n = np.zeros((1024, 1920, 3), dtype=np.uint8)
    warp_mat=getSimilarityTransform_fast(source_pose,target_pose)
    trans_pose_points = transPoints(source_points.pose_points, warp_mat)
    showPoints(black_s, trans_pose_points,(255,0,255),0)

    tmp=args.frameSize
    target_frameSize=(int(tmp.split('x')[1]),int(tmp.split('x')[0]))

    out_video_path=args.out_video
    out=cv2.VideoWriter(out_video_path,cv2.VideoWriter_fourcc(*'XVID'),25,target_frameSize)

    step = 0
    done = False
    total_steps=0
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    keyPoints_json_paths = sorted(make_dataset(args.source_points_dir))
    windows = deque()
    for p in keyPoints_json_paths:
        print(p)
        black=np.zeros((1024,1920,3),dtype=np.uint8)
        points = get_points(p)
        if points.valid==-1:
            continue
        showPoints(black, points.pose_points, (0, 0, 255),-1)
        trans_pose_points=transPoints(points.pose_points,warp_mat)
        #showPoints(black, trans_pose_points, (0, 0, 255), 5)
        trans_face_points = transPoints(points.face_points, warp_mat)
        trans_hand_left_points = transPoints(points.hand_left_points, warp_mat)
        trans_hand_right_points = transPoints(points.hand_right_points, warp_mat)

        trans_points=keyPoints(pose_points=trans_pose_points, face_points=trans_face_points,
                               hand_left_points=trans_hand_left_points,
                               hand_right_points=trans_hand_right_points, valid=1)
        windows.append(trans_points)

        if len(windows) > args.win_size:
            windows.popleft()
        if len(windows) == args.win_size or True:
            points_smoothed = getMedianPoints(windows)
            tmp_image=np.copy(refer_image)
            showPoints(tmp_image, points_smoothed.pose_points, (0, 255, 255), -1)
            out.write(tmp_image)
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
            out_path = os.path.join(args.out_dir, p_name.split('.')[0] + ".txt")
            with open(out_path, 'w') as fout:
                fout.write(pose_str + '\n')
                fout.write(face_str + '\n')
                fout.write(hand_left_str + '\n')
                fout.write(hand_right_str)