import math
import pickle
from .kalmanfilter import KalmanFilter1D, KalmanFilterND
from .support_funcs import *


vert_border = 10
horizontal_border = 10


def kfcalc(kalmanfilter, z):
    kalmanfilter.predict()
    kalmanfilter.update(z)
    return kalmanfilter.x


def warp(frame, affine, size):
    frame = cv2.warpAffine(frame, affine, size)
    frame = cut(vert_border, horizontal_border, frame)
    return frame


def compensating_transform(original, new):
    from numpy.linalg import inv
    A = np.dot(new[:2, :2], inv(original[:2, :2]))
    b = np.dot(-A, original[:2, 2:]) + new[:2, 2:]
    A = inv(A)
    b = np.dot(-A, b)
    affine = np.insert(A, [2], b, axis=1)
    return affine


def sum_2_affine(a1, a2):
    A1, b1 = a1[:2, :2], a1[:2, 2:]
    A2, b2 = a2[:2, :2], a2[:2, 2:]
    A = np.dot(A2, A1)
    b = np.dot(A2, b1) + b2
    affine = np.insert(A, [2], b, axis=1)
    return affine


def video_stab(filename, new_size=(640, 480), tracking_mode=True):
    """
    Simple video live stabilization via recursive Kalman Filter
    :param filename: path to video file
    :param new_size: new (width, height) size of video
    :param tracking_mode: set True to show tracking  of key points
    :return: None
    """
    cov = filename.split('/')[1][:-4]
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
    with open(cov+'_covariance1.pickle', 'rb') as file:
        R = pickle.load(file)
    Q, P = np.diag([1e-7, 1e-7, 4e-3, 1e-7, 1e-7, 4e-3]), np.eye(6)
    F, H = np.eye(6), np.eye(6)
    X = np.zeros((6, 1))
    kf_6 = KalmanFilterND(X, F, H, P, Q, R)
    # -----------------------------------------------------------------
    with open(cov+'_covariance2.pickle', 'rb') as file:
        R = pickle.load(file)
    Q, P = np.diag([1e-7, 1e-7]), np.eye(2)
    H = np.eye(2)
    F = np.eye(2)
    X = np.zeros((2, 1))
    kf_2 = KalmanFilterND(X, F, H, P, Q, R)
    # ------------------------------------------------------------------
    with open(cov+'_covariance3.pickle', 'rb') as file:
        R = pickle.load(file)
    F = np.eye(3)
    H = np.eye(3)
    X = np.zeros(3)
    P = np.ones(3)
    Q = np.diag([4e-3, 4e-3, 1e-6])
    kf_3 = KalmanFilterND(X, F, H, P, Q, R)
    # ------------------------------------------------------------------
    '''kfs = [KalmanFilter1D(), KalmanFilter1D(), KalmanFilter1D(),
           KalmanFilter1D(), KalmanFilter1D(), KalmanFilter1D()]
    for k in kfs:
        k.x = 0
        k.covariance_P = 1
        k.covariance_R = 0.1**2
        k.covariance_Q = 0.0001'''
    # ------------------------------------------------------------------
    cap, n_frames, fps, prev = video_open(filename, new_size)

    old, smoothed_affine, smoothed_translational, smoothed_similarity = [], [], [], []
    last_affine = ...
    # video writer args
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(5)
    video_stab = filename[:-4] + 'stab.mp4'
    out = cv2.VideoWriter(video_stab, fourcc, fps, new_size)
    cumulative_transform = np.insert(np.array([[1, 0], [0, 1]]), [2], [0], axis=1)
    cumulative_smoothed1 = cumulative_transform.copy()
    cumulative_smoothed2 = cumulative_transform.copy()
    cumulative_smoothed3 = cumulative_transform.copy()
    for i in range(n_frames-1):
        # read frames
        ret2, cur = cap.read()
        cur = cv2.resize(cur, new_size, cv2.INTER_CUBIC)
        # get affine transform between frames
        affine = cv2.estimateRigidTransform(prev, cur, False)
        # Sometimes there is no Affine transform between frames, so we use the last
        if not np.all(affine):
            affine = last_affine
        last_affine = affine
        # Accumulated frame to frame original transform
        cumulative_transform = sum_2_affine(cumulative_transform, affine)
        # save original affine for comparing with stabilized
        old.append(cumulative_transform)
        z = np.array([affine.ravel()]).T    # (a1, a2, b1, a3, a4, b2)^T
        z1 = affine[:2, 2:]  # b1, b2
        z2 = affine[0][2], affine[1][2], math.atan2(affine[1][0], affine[0][0])  # (b1, b2, a)
        # predict new vector and update
        x1 = kfcalc(kf_6, z)
        x2 = kfcalc(kf_2, z1)
        x3 = kfcalc(kf_3, z2)
        # create new Affine transform

        smoothed_affine_motion = np.float32(x1.reshape(2, 3))
        affine_motion = compensating_transform(smoothed_affine_motion, cumulative_transform)

        a11, a22 = math.cos(x3[2]), math.sin(x3[2])
        smoothed_similarity_motion = np.array([[a11, -a22, x3[0]], [a22, a11, x3[1]]])
        similarity_motion = compensating_transform(smoothed_similarity_motion, cumulative_transform)

        smoothed_translational_motion = np.array([[1, 0, x2[0]], [0, 1, x2[1]]])
        translational_motion = compensating_transform(smoothed_translational_motion, cumulative_transform)



        # get stabilized frame
        cur1 = warp(cur, affine_motion, new_size)
        cur2 = warp(cur, translational_motion, new_size)
        cur3 = warp(cur, similarity_motion, new_size)
        if i > 1 and tracking_mode:
            tr1, tr2 = tracked[0](prev, cur), tracked[1](prev1, cur1)
            tr3, tr4 = tracked[2](prev2, cur2), tracked[3](prev3, cur3)
        else:
            tr1, tr2, tr3, tr4 = cur, cur1, cur2, cur3
        # Accumulated frame to frame smoothed transform
        # smoothed cumulative transform affine model
        cumulative_smoothed1 = sum_2_affine(cumulative_smoothed1, smoothed_affine_motion)
        smoothed_affine.append(cumulative_smoothed1)
        # smoothed cumulative transform similarity model
        cumulative_smoothed2 = sum_2_affine(cumulative_smoothed2, smoothed_similarity_motion)
        smoothed_similarity.append(cumulative_smoothed2)
        # smoothed cumulative transform translational model
        cumulative_smoothed3 = sum_2_affine(cumulative_smoothed3, smoothed_translational_motion)
        smoothed_translational.append(cumulative_smoothed3)
        # concatenate original and stabilized frames
        result = concatenate_n_images(tr1, tr2, tr3, tr4)
        cv2.imshow('Original/stab1/stab2', result)
        out.write(tr2)
        prev, prev1, prev2, prev3 = tr1, tr2, tr3, tr4
        if cv2.waitKey(np.int(1000//fps)) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # plot affine transform params trajectories

    trajectory(old, 'r')
    trajectory(smoothed_affine, 'g')
    trajectory(smoothed_similarity, 'b')
    trajectory(smoothed_translational, 'y')
    '''pickle.dump(covariance(*get_params_from_trajectory(old)), open('jitter4_covariance1.pickle', 'wb'))
    p = get_params_from_trajectory(old)
    pickle.dump(covariance(p[2], p[5]), open('jitter4_covariance2.pickle', 'wb'))
    pickle.dump(covariance(p[2], p[5], np.arctan2(p[3], p[0])), open('jitter4_covariance3.pickle', 'wb'))'''
    plt.show()

