import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import pickle
from .kalmanfilter import KalmanFilterND, KalmanFilter1D
from .support_funcs import *

vert_border = 10
horizontal_border = 10


def video_stab(filename, new_size=(640, 480), tracking_mode=True):
    """
    Simple video live stabilization via recursive Kalman Filter
    :param filename: path to video file
    :param new_size: new (width, height) size of video
    :param tracking_mode: set True to show tracking  of key points
    :return: None
    """
    if tracking_mode:
        from .curve import tracking

        def decorator(func):
            funcs = {}
            for i in range(4):
                @tracking(track_len=20, detect_interval=10)
                def f(prev, cur):
                    return func(prev, cur)
                funcs[i] = f
            return funcs

        @decorator
        def tracked(prev, cur):
            return get_grey_images(prev, cur)

    print('Video ' + filename + ' processing')
    with open('videostab/covariance.pickle', 'rb') as file:
        R = pickle.load(file)
    Q, P = np.diag([1e-7, 1e-7, 4e-3, 1e-7, 1e-7, 4e-3]), np.diag([1, 1, 1, 1, 1, 1])
    F, H = np.eye(6), np.eye(6)
    X = np.zeros((6, 1))
    kf_6 = KalmanFilterND(X, F, H, P, Q, R)     # multivariate 5 order Kalman Filter
    # Noispection PeP8
    F, H = np.eye(3), np.eye(3)
    X = np.zeros(3)
    P = np.ones(3)
    Q = np.diag([4e-3, 4e-3, 1e-6])
    with open('videostab/covariance_3.pickle', 'rb') as file:
        R = pickle.load(file)
    #R = np.diag([0.25, 0.25, 0.25])**2
    kf_3 = KalmanFilterND(X, F, H, P, Q, R)  # invariate 2 order Kalman Filter

    kfs = [KalmanFilter1D(), KalmanFilter1D(), KalmanFilter1D(),
           KalmanFilter1D(), KalmanFilter1D(), KalmanFilter1D()]
    for k in kfs:
        k.x = 0
        k.covariance_P = 1
        k.covariance_R = 0.1**2
        k.covariance_Q = 0.0001

    cap, n_frames, fps, prev = video_open(filename, new_size)

    old, new_6, new_3, new_6_ = [], [], [], []
    last_affine = ...
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
        # predict new state
        kf_6.predict()
        kf_3.predict()
        for k in kfs:
            k.predict()
        z = np.array([affine.ravel()]).T    # (a1, a2, b1, a3, a4, b2)^T
        d = affine[0][2], affine[1][2], math.atan2(affine[1][0], affine[0][0])  # (b1, b2, a)
        # update
        kf_6.update(z)
        kf_3.update(d)
        for p, k in zip(z.tolist(), kfs):
            k.update(p[0])
        # get state vector
        x = [k.x for k in kfs]
        X = kf_6.x
        d = kf_3.x
        # create new Affine transform
        new_trans_6_ = np.array(x).reshape(2, 3)
        new_6_.append(new_trans_6_)
        new_trans_6 = np.float32(X.reshape(2, 3))
        a11, a22 = math.cos(d[2]), math.sin(d[2])
        new_trans_3 = np.array([[a11, -a22, d[0]],
                               [a22, a11, d[1]]])
        # get stabilized frame
        cur2 = cv2.warpAffine(prev, new_trans_6, new_size)
        cur3 = cv2.warpAffine(prev, new_trans_3, new_size)
        cur4 = cv2.warpAffine(prev, new_trans_6_, new_size)
        # crop borders
        cur2 = cut(vert_border, horizontal_border, cur2)
        cur3 = cut(vert_border, horizontal_border, cur3)
        cur4 = cut(vert_border, horizontal_border, cur4)
        if i > 1 and tracking_mode:
            tr1, tr2 = tracked[0](prev, cur), tracked[1](prev2, cur2)
            tr3, tr4 = tracked[2](prev3, cur3), tracked[3](prev4, cur4)
        else:
            tr1, tr2, tr3, tr4 = cur, cur2, cur3, cur4
        # save transforms for analysis
        new_6.append(new_trans_6)
        new_3.append(new_trans_3)
        # concatenate original and stabilized frames
        result = concatenate_n_images(tr1, tr2, tr3, tr4)
        cv2.imshow('Original/stab1/stab2', result)
        out.write(cur2)
        prev, prev2, prev3, prev4 = cur, cur2, cur3, cur4
        if cv2.waitKey(np.int(1000//fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # plot affine transform params trajectories
    trajectory(old, 'r')
    trajectory(new_6, 'g')
    trajectory(new_3, 'b')
    trajectory(new_6_, 'm')
    plt.show()


