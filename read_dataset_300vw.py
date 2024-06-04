import cv2
import numpy as np
import sys
from os import path
from glob import glob
import utils
import random

# Dataset format
# 300VW
# ├── testset
# │   ├── sample name
# │   │   ├── annot / {frame_no}.pts
# │   │   └── vid.avi
# └── trainset
#     ├── sample name
#     │   ├── annot / {frame_no}.pts
#     │   └── vid.avi
# frame_no format "%06d"

# left eye, right eye and mouth points each with (left, right) pairs


class Video_300VW:
    six_key_points_idxs = [36, 39, 42, 45, 48, 54]

    def __init__(self, vid_path, n_points):
        assert n_points == 68 or n_points == 6, "Invalid n_points: it should either be 68 or 6"
        self.vid = cv2.VideoCapture(vid_path)
        self.f_count = 0
        self.annot_dir = path.join(path.dirname(vid_path), "annot")
        self.n_points = n_points
        self.vid_name = vid_path.split(path.sep)[-2]
        self.done = False

    def next_frame(self):

        if self.done: return False, None, None, None, None

        got_frame, img = self.vid.read()
        if not got_frame:
            self.vid.release()
            self.done = True
            return False, None, None, None, None

        self.f_count += 1
        marks = utils.read_pts_file(path.join(self.annot_dir, "%06d.pts"%self.f_count))
        bbox = utils.bbox_from_marks(marks)
        if self.n_points == 6: marks = marks[Video_300VW.six_key_points_idxs, :]
        # sample_name = vid_name + f_count
        sample_name = self.vid_name + "_%06d"%self.f_count

        return True, sample_name, img, bbox, marks

# read video with annot
def read_video_with_annot(vid_path, n_points = 68):
    """
    Generator for reading video and annotation
    Args:
        vid_path: Path of the video file
    Return:
        img, marks
    """
    vid = Video_300VW(vid_path, n_points)
    while True:
        got_frame, sample_name, img, bbox, marks = vid.next_frame()
        if not got_frame: break
        yield sample_name, img, bbox, marks
    return

# display video with ann
def show_video_with_ann(vid_path, n_points):
    f_count = 0
    vid_name = path.split(vid_path)[-2]
    for _, img, bbox, marks in read_video_with_annot(vid_path, n_points = n_points):
        f_count += 1

        s_img = np.copy(img)

        utils.draw_bbox(s_img, bbox)
        # max bbox
        utils.draw_bbox(s_img, utils.scale_bbox(bbox, 1.2), (0,255,0))
        utils.draw_marks(s_img, marks, draw_idx = True)
        cv2.imshow(path.basename(vid_path), s_img)

        # testing scale_bbox_and_crop
        img, _, marks = utils.scale_bbox_and_crop(img, bbox, marks,
                                                           np.random.uniform(1, 1.2))
        img = utils.apply_img_hist(img)
        img, marks = utils.resize_to_input_shape(img, marks, (128, 128))
        ss_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        utils.draw_marks(ss_img, marks, draw_idx = True)
        cv2.imshow("crop", ss_img)

        # testing augment_flip
        flip_img, flip_marks = utils.augment_flip(img, marks)
        s_flip_img = cv2.cvtColor(flip_img, cv2.COLOR_GRAY2BGR)
        utils.draw_marks(s_flip_img, flip_marks, draw_idx = True)
        cv2.imshow("flip", s_flip_img)

        if utils.opencv_pause_quit_window(): break

    print(f"Video:{vid_name} Frame_count: {f_count}")

if __name__ == "__main__":

    random.seed(123)
    np.random.seed(123)

    assert len(sys.argv) == 2, "Pass the 300vw datasetdir as command line arg"

    files_list = glob(path.join(sys.argv[1], "*", "*", "vid.avi"))
    print(f"n files:{len(files_list)}")

    n_points = 6
    while True:
        show_video_with_ann(random.choice(files_list), n_points)
