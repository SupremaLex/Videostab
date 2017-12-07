# file videostab.py
import math
from .kalmanfilter import KalmanFilterND
from .support_funcs import *
from .videostab import warp


def demostrating_video_stab(filename, new_size=(320, 240), tracking_mode=True):
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
                @tracking(track_len=20, detect_interval=5)
                def f(prev, cur):
                    return func(prev, cur)
                funcs[i] = f
            return funcs

        @decorator
        def tracked(prev, cur):
            return get_grey_images(prev, cur)

    print('Video ' + filename + ' processing')
    R = get_cov_from_video(filename, new_size)*1e-2
    Q, P = np.diag([1e-8, 1e-7, 4e-3, 1e-7, 1e-8, 4e-3]), np.eye(6)
    F, H = np.eye(6), np.eye(6)
    X = np.zeros((6, 1))
    kf_6 = KalmanFilterND(X, F, H, P, Q, R)
    # -----------------------------------------------------------------
    R = np.ones((2, 2))*1e-6
    Q, P = np.diag([1e-3, 1e-3]), np.eye(2)
    H = np.eye(2)
    F = np.eye(2)
    X = np.zeros((2, 1))
    kf_2 = KalmanFilterND(X, F, H, P, Q, R)
    # ------------------------------------------------------------------
    R = np.ones((3, 3))*1e-6
    F = np.eye(3)
    H = np.eye(3)
    X = np.zeros(3)
    P = np.ones(3)
    Q = np.diag([4e-3, 4e-3, 1e-7])
    kf_3 = KalmanFilterND(X, F, H, P, Q, R)
    # ------------------------------------------------------------------
    cap, n_frames, fps, prev = video_open(filename, new_size)

    old, smoothed_affine, smoothed_translational, smoothed_similarity = [], [], [], []
    # video writer args
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(5)
    video_stab = filename[:-4] + 'stab.mp4'
    out = cv2.VideoWriter(video_stab, fourcc, fps, new_size)
    cumulative_transform = np.insert(np.array([[1, 0], [0, 1]]), [2], [0], axis=1)
    last_affine = cumulative_transform.copy()
    cumulative_smoothed1 = cumulative_transform.copy()
    cumulative_smoothed2 = cumulative_transform.copy()
    cumulative_smoothed3 = cumulative_transform.copy()
    for i in range(n_frames-1):
        # read frames
        ret2, cur = cap.read()
        cur = cv2.resize(cur, new_size, cv2.INTER_AREA)
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
        x1 = kf_6.predict_and_update(z)
        x2 = kf_2.predict_and_update(z1)
        x3 = kf_3.predict_and_update(z2)

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
        cv2.imshow('Original/smoothed', result)
        out.write(tr2)
        prev, prev1 = tr1, tr2
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

    plt.show()

