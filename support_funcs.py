from matplotlib import pyplot as plt
import cv2
import numpy as np


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
    :return:
    """
    number = len(trajectory)
    params = ('a1', 'a2', 'b1', 'a3', 'a4', 'b2')
    trajectories = dict(zip(params, get_params_from_trajectory(trajectory)))
    frames = range(number)
    for k in trajectories:
        f = plt.figure(k)
        plt.plot(frames, trajectories[k], figure=f, color=color)
        plt.xlabel('frame', figure=f)
        plt.ylabel(k, figure=f)


def rotate_video(videoname, angle):
    """
    Rotate video by angle (clockwise)
    :param videoname: path to video file
    :param angle: angle (clockwise)
    :return: None, but rotated video is saved as 'videonamerot.mp4'
    """
    file = videoname
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


def video_open(videoname, size=(640, 480)):
    """
    :param videoname: path to video file
    :param size: new size of video
    :return: VideoCapture object, number of frames, fps, and first frame
    """
    cap = cv2.VideoCapture(videoname)
    nframes = np.int(cap.get(7))
    fps = cap.get(5)
    ret, prev = cap.read()
    prev = cv2.resize(prev, size, cv2.INTER_CUBIC)
    return cap, nframes, fps, prev


def show(videoname, size=(640, 480), tracking_mode=False):
    """
    Just show resized video
    :param videoname: path to video file
    :param size: new (width, height) of image
    :param tracking_mode: set True to show tracking  of key points
    :return: None
    """
    cap, nframes, fps, prev = video_open(videoname, size)
    if tracking_mode:

        from curve import tracking

        @tracking(track_len=10, detect_interval=10)
        def tracked(prev, cur):
            return get_grey_images(prev, cur)

    for i in range(nframes-2):
        ret, cur = cap.read()
        cur = cv2.resize(cur, size, cv2.INTER_CUBIC)
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
