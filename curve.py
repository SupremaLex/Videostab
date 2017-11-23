import cv2
import numpy as np


# params for Shi-Tomasi corner detection
feature_params = dict(maxCorners=50,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Kanade-Lucas optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def tracking(track_len=10, detect_interval=5):
    """
    Decorator, which find features and tracking them
    For use :
    1. Create decorator object with settings you need, like this tracking(track_len=15, detect_interval=10)
    2. Use created decorator class to warp function, which return two serial frames
    :param track_len: max length for tracking list for each key point
    :param detect_interval: detect interval means, that we find new features every *detect_interval* frames
    :return: decorator class Curve
    """
    class Curve:

        def __init__(self, func):
            self.tracks = []
            self.function = func
            self.frame_idx = 0

        def __call__(self, prev, cur):
            prev_grey, cur_grey = self.function(prev, cur)
            if len(prev_grey.shape) != 2:
                prev_grey = cv2.cvtColor(prev_grey, cv2.COLOR_BGR2GRAY)
            if len(cur_grey.shape) != 2:
                cur_grey = cv2.cvtColor(cur_grey, cv2.COLOR_BGR2GRAY)
            vis = cur.copy()

            if len(self.tracks) > 0:
                img0, img1 = prev_grey, cur_grey
                # get the last keypoints
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                # find corresponding features in prev -> cur and cur -> prev optical flow "directions"
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                # create list of bool values
                good = d < 1
                new_tracks = []
                # for each feature in list, add a next feature
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    # pop first element if we have more then track_len features
                    if len(tr) > track_len:
                        tr.pop(0)
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (255, 0, 0), -1)
                self.tracks = new_tracks
                # draw
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (255, 0, 0))

            if self.frame_idx % detect_interval == 0:
                mask = np.zeros_like(cur_grey)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                # get features Shi-Tomasi corner detection algorithm
                p = cv2.goodFeaturesToTrack(cur_grey, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = cur_grey
            return vis

    return Curve
