from .support_funcs import *
import cv2


def mosaic(filename, cumulative, compensating, new_size, m):
    cap, n_frames, fps, prev = video_open(filename, new_size)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = cap.get(5)
    video_stab = filename[:-4] + 'mosaicked.mp4'
    out = cv2.VideoWriter(video_stab, fourcc, fps, new_size)
    frames = []
    for i in range(n_frames-1):
        r, frame = cap.read()
        frame = cv2.resize(frame, new_size, cv2.INTER_AREA)
        frames.append(frame)

    for i in range(m+1, len(frames[:-m])):
        group = []
        center = i//2
        for j, f in enumerate(frames[i-m:i+m]):
            if j != m+1:
                if j < m+1:
                    cum = cumulative[center-j]
                else:
                    cum = cumulative[center+j]
                affine = compensating_transform(cum, compensating[center])[0]
                warped = cv2.warpAffine(frames[center], affine, new_size)
                group.append(warped)
        imgs = [frames[center]]+group
        mosaicked_cur = combine_n_images(*imgs)
        #out.write(warped_cur)
        img = concatenate_n_images(frames[i], mosaicked_cur)
        cv2.imshow('mosaicking', img)
        if cv2.waitKey(np.int(1000//fps)) & 0xFF == ord('q'):
            break