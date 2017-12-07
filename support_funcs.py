from matplotlib import pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as mpatches


def get_params_from_trajectory(trajectory):
    """
    Get trajectory of each affine model params separately
    :param trajectory: affine transform matrix (2x3 dims)
    :return: list of list params
    """
    params = [[], [], [], [], [], []]
    for e in trajectory:
        params[0].append(e[0][0])  # a1
        params[1].append(e[0][1])  # a2
        params[2].append(e[0][2])  # b1
        params[3].append(e[1][0])  # a3
        params[4].append(e[1][1])  # a4
        params[5].append(e[1][2])  # b2
    return params


def trajectory(trajectory, color='r'):
    """
    Draw plots for all pairs matching params of old and new trajectory, such as old a1 vs new a2 etc.
    :param trajectory: affine transform matrix (2x3 dims)
    :param color: color for plotting
    :return: None
    """
    number = len(trajectory)
    params = ('a1', 'a2', 'b1', 'a3', 'a4', 'b2')
    trajectories = dict(zip(params, get_params_from_trajectory(trajectory)))
    frames = range(number)
    red_patch = mpatches.Patch(color='r', label='original motion')
    green_patch = mpatches.Patch(color='g', label='affine motion')
    blue_patch = mpatches.Patch(color='b', label='similarity motion')
    yellow_patch = mpatches.Patch(color='y', label='translational motion')
    for k in trajectories:
        f = plt.figure(k)
        plt.plot(frames, trajectories[k], figure=f, color=color, label=k)
        plt.xlabel('frame', figure=f)
        plt.ylabel(k, figure=f)
        plt.legend(handles=[red_patch, green_patch, blue_patch, yellow_patch])


def rotate_video(video_name, angle):
    """
    Rotate video by angle (clockwise)
    :param video_name: path to video file
    :param angle: angle (clockwise)
    :return: None, but rotated video is saved as 'video_namerot.mp4'
    """
    file = video_name
    cap = cv2.VideoCapture(file)
    nframes = np.int(cap.get(7))
    s1, s2 = np.int(cap.get(3)), np.int(cap.get(4))
    # videowriter args
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(5)
    videostab = file[:-4] + 'rot.mp4'
    out = cv2.VideoWriter(videostab, fourcc, fps, (s2, s1))
    rot = cv2.getRotationMatrix2D((s1 // 2,  s2 // 2), -angle, 1)

    for i in range(nframes-1):
        r, frame = cap.read()
        frame = cv2.warpAffine(frame, rot, (s2, s1))
        out.write(frame)
    cap.release()
    out.release()


def get_grey_images(img1, img2):
    """
    Just get grey images from colored images
    :param img1: colored img1
    :param img2: colored img2
    :return: grey img1, grey img2
    """
    img1_grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_grey = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1_grey, img2_grey


def video_open(video_name, size=(640, 480)):
    """
    :param video_name: path to video file
    :param size: new size of video
    :return: VideoCapture object, number of frames, fps, and first frame
    """
    cap = cv2.VideoCapture(video_name)
    nframes = np.int(cap.get(7))
    fps = cap.get(5)
    ret, prev = cap.read()
    prev = cv2.resize(prev, size, cv2.INTER_CUBIC)
    return cap, nframes, fps, prev


def show(video_name, size=(640, 480), tracking_mode=False):
    """
    Just show resized video
    :param video_name: path to video file
    :param size: new (width, height) of image
    :param tracking_mode: set True to show tracking  of key points
    :return: None
    """
    cap, nframes, fps, prev = video_open(video_name, size)
    if tracking_mode:

        from .curve import tracking

        @tracking(track_len=20, detect_interval=10)
        def tracked(prev, cur):
            return get_grey_images(prev, cur)

    for i in range(nframes-2):
        ret, cur = cap.read()
        cur = cv2.resize(cur, size, cv2.INTER_CUBIC)
        if tracking_mode:
            cur = tracked(prev, cur)
        cv2.imshow('show', cur)
        if cv2.waitKey(np.int(1000//fps)) & 0xFF == ord('q'):
            break
        prev = cur


def concatenate_n_images(*args):
    """
    Concatenate n images side-by-side
    :param args: n images
    :return: concatenated side-by-side images
    """
    max_height = 0
    total_width = 0
    for im in args:
        max_height = max(max_height, im.shape[0])
        total_width += im.shape[1]
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.uint8)
    h, next_ind = 0, 0
    for im in args:
        h = im.shape[0]
        new_img[:h, next_ind: next_ind + im.shape[1]] = im
        next_ind += im.shape[1]
    return new_img


def cut(vertical, horizontal, img):
    """
    Cut image
    :param vertical: vertical border to cut
    :param horizontal: horizontal border to cut
    :param img: img
    :return: cut img
    """
    size = img.shape[:2][::-1]
    img = img[vertical: -vertical, horizontal: -horizontal]
    img = cv2.resize(img, size, cv2.INTER_CUBIC)
    return img


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


def covariance(*series):
    R = np.cov(series)
    return R


def get_cov_from_video(video_name):
    cap, n_frames, fps, prev = video_open(video_name, size=(640, 480))
    new_size = 640, 480
    old = []
    last_affine = ...
    cumulative_transform = np.insert(np.array([[1, 0], [0, 1]]), [2], [0], axis=1)
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
    cov = covariance(*get_params_from_trajectory(old))
    return cov
