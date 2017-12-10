from .kalmanfilter import KalmanFilterND
from .support_funcs import *
from .mosaicking import mosaic


def warp(frame, affine, crop, size):
    horizontal_border = crop
    frame = cv2.warpAffine(frame, affine, size)
    vert_border = horizontal_border * size[0]//size[1]
    #frame = cut(vert_border, horizontal_border, frame)
    return frame


def video_stab(filename, new_size=(320, 240), crop=25, tracking_mode=True):
    """
    Simple video live stabilization via recursive Kalman Filter
    :param filename: path to video file
    :param new_size: new (width, height) size of video
    :param crop: crop *crop* number of colons
    :param tracking_mode: set True to show tracking  of key points
    :return: None
    """
    if tracking_mode:
        from .curve import tracking

        @tracking(track_len=20, detect_interval=5)
        def tracked(prev, cur):
            return get_grey_images(prev, cur)

    print('Video ' + filename + ' processing')
    R = get_cov_from_video(filename, new_size)*1e-2
    Q, P = np.diag([1e-8, 1e-7, 4e-3, 1e-7, 1e-8, 4e-3]), np.eye(6)
    F, H = np.eye(6), np.eye(6)
    X = np.zeros((6, 1))
    kf = KalmanFilterND(X, F, H, P, Q, R)
    cap, n_frames, fps, prev = video_open(filename, new_size)

    old, smoothed_affine, smoothed_translational, smoothed_similarity = [], [], [], []
    # video writer args
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(5)
    new_name = filename[:-4] + 'stab.mp4'
    out = cv2.VideoWriter(new_name, fourcc, fps, new_size)
    cumulative_transform = np.insert(np.array([[1, 0], [0, 1]]), [2], [0], axis=1)
    last_affine = cumulative_transform.copy()
    cumulative_smoothed1 = cumulative_transform.copy()
    compesating = []
    for i in range(n_frames-1):
        # read frames
        ret2, cur = cap.read()
        cur = cv2.resize(cur, new_size, cv2.INTER_AREA)
        # get affine transform between frames
        affine = cv2.estimateRigidTransform(prev, cur, True)
        # Sometimes there is no Affine transform between frames, so we use the last
        if not np.all(affine):
            affine = last_affine
        last_affine = affine
        # Accumulated frame to frame original transform
        cumulative_transform = sum_2_affine(cumulative_transform, affine)
        # save original affine for comparing with stabilized
        old.append(cumulative_transform)
        z = np.array([affine.ravel()]).T    # (a1, a2, b1, a3, a4, b2)^T
        # predict new vector and update
        x1 = kf.predict_and_update(z)

        # create new Affine transform

        smoothed_affine_motion = np.float32(x1.reshape(2, 3))
        affine_motion, b = compensating_transform(smoothed_affine_motion, cumulative_transform)
        compesating.append(b)
        # get stabilized frame
        cur1 = warp(cur, affine_motion, crop, new_size)
        if i > 1 and tracking_mode:
            tr1, tr2 = cur, tracked(prev1, cur1)
        else:
            tr1, tr2 = cur, cur1
        # Accumulated frame to frame smoothed transform
        # smoothed cumulative transform affine model
        cumulative_smoothed1 = sum_2_affine(cumulative_smoothed1, smoothed_affine_motion)
        smoothed_affine.append(cumulative_smoothed1)
        out.write(tr2)
        prev, prev1 = tr1, tr2

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    mosaic(new_name, old, compesating, new_size, m=2)

