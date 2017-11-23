import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import pickle
from kalmanfilter import KalmanFilter
from support_funcs import *

vert_border = 10
horizontal_border = 10


def videostab(filename, new_size=(640, 480), tracking_mode=True):
    """
    Simple video live stabilization via recursive Kalman Filter
    :param filename: path to video file
    :param new_size: new (width, height) size of video
    :param tracking_mode: set True to show tracking  of key points
    :return: None
    """
    if tracking_mode:
        from curve import tracking

        @tracking(track_len=10, detect_interval=10)
        def tracked(prev, cur):
            return get_grey_images(prev, cur)
    print('Video ' + filename + ' processing')
    with open('covariance.pickle', 'rb') as file:
        R = pickle.load(file)
    Q, P = np.diag([1e-6, 1e-6, 4e-3, 1e-6, 1e-6, 5e-1]), np.diag([1, 1, 1, 1, 1, 1])
    F, H = np.eye(6), np.eye(6)
    X = np.zeros((6, 1))
    kf_6 = KalmanFilter(X, F, H, P, Q, R)

    F, H = np.eye(3), np.eye(3)
    X = np.zeros(3)
    P = np.ones(3)
    Q = np.array([4e-3, 5e-3, 1e-4])
    R = np.array([4e-4, 4e-4, 1e-5])*100
    kf_3 = KalmanFilter(X, F, H, P, Q, R, 1)

    cap, n_frames, fps, prev = video_open(filename, new_size)

    old, new_6, new_3 = [], [], []
    # video writer args
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(5)
    video_stab = filename[:-4] + 'stab.mp4'
    out = cv2.VideoWriter(video_stab, fourcc, fps, new_size)
    for i in range(n_frames-1):
        # read frames
        ret2, cur = cap.read()
        cur = cv2.resize(cur, new_size, cv2.INTER_CUBIC)
        # get affine transform between frames
        affine = cv2.estimateRigidTransform(prev, cur, True)
        # Sometimes there is no Affine transform between frames, so we use the last
        if not np.all(affine):
            affine = last_affine
        last_affine = affine
        # save original affine for comparing with stabilized
        old.append(affine)
        kf_6.predict()
        z = np.array([affine.ravel()]).T
        kf_6.update(z)
        X = kf_6.x
        new_trans_6 = np.float32(X.reshape(2, 3))
        # b1, b2, a
        d = affine[0][2], affine[1][2], math.atan2(affine[1][0], affine[0][0])
        kf_3.predict()
        kf_3.update(d)
        # create new Affine transform
        d = kf_3.d
        a11 = math.cos(d[2])
        a22 = math.sin(d[2])
        new_trans_3 = np.array([[a11, -a22, d[0]],
                               [a22, a11, d[1]]])
        # get stabilized frame
        cur2 = cv2.warpAffine(prev, new_trans_6, new_size)
        cur3 = cv2.warpAffine(prev, new_trans_3, new_size)
        # crop borders
        cur2 = cut(vert_border, horizontal_border, cur2)
        cur3 = cut(vert_border, horizontal_border, cur3)
        if i > 1 and tracking_mode:
            tr1, tr2, tr3 = tracked(prev, cur), tracked(prev2, cur2), tracked(prev3, cur3)
        else:
            tr1, tr2, tr3 = cur, cur2, cur3
        new_6.append(new_trans_6)
        new_3.append(new_trans_3)
        # concatenate original and stabilized frames
        result = concatenate_n_images(tr1, tr2, tr3)
        cv2.imshow('Original/stab1/stab2', result)
        out.write(cur2)
        prev, prev2, prev3 = cur, cur2, cur3
        if cv2.waitKey(np.int(1000//fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    trajectory(old, 'r')
    trajectory(new_6, 'g')
    trajectory(new_3, 'b')
    plt.show()
    return videostab


if __name__ == '__main__':
    name = videostab('Video0009.mp4', (320, 240), tracking_mode=False)

