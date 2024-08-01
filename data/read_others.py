import cv2
import numpy as np
from os import path
from glob import glob
import sys
from . import utils

# Dataset format
# "dataset"
# ├── testset
# │   ├── {img}.(jpg|png)
# │   ├── {img}.pts
# └── trainset
#     ├── {img}.(jpg|png)
#     ├── {img}.pts

def read_img_with_annot(img_path, n_points = 68):
    img = cv2.imread(img_path)
    marks = utils.read_pts_file(path.splitext(img_path)[0] + ".pts")
    bbox = utils.bbox_from_marks(marks)
    assert n_points in utils.marks_map_68.keys(), f"Map for {n_points}pts not found"
    marks = marks[utils.marks_map_68[n_points], :]
    # get sample name - remove extension
    sample_name = path.basename(img_path)[:-4]
    return sample_name, img, bbox, marks

def show_img_with_ann(img_paths_l, n_points = 68):
    for i, img_path in enumerate(img_paths_l):
        print(f"{i+1}/{len(img_paths_l)} {img_path}")

        _, img, bbox, marks = read_img_with_annot(img_path, n_points)

        s_img = np.copy(img)
        utils.draw_bbox(s_img, bbox)
        # max bbox
        utils.draw_bbox(s_img, utils.scale_bbox(bbox, 1.2), (0,255,0))
        utils.draw_marks(s_img, marks, draw_idx = True)
        cv2.imshow("Show", cv2.resize(s_img, (640, 480)))

        # testing scale_bbox_and_crop
        img, bbox, marks = utils.scale_bbox_and_crop(img, bbox, marks,
                                                       np.random.uniform(1, 1.2))
        img = utils.apply_img_hist(img)
        img, marks = utils.resize_to_input_shape(img, marks, (128, 128))
        marks = np.round(marks).astype(np.int32)
        assert np.all(marks >= 0) and np.all(marks <= 128), "marks out of bounds"
        ss_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        utils.draw_marks(ss_img, marks, draw_idx = True)
        cv2.imshow("crop", ss_img)

        # testing augment_flip
        flip_img, flip_marks = utils.augment_flip(img, marks)
        s_flip_img = cv2.cvtColor(flip_img, cv2.COLOR_GRAY2BGR)
        utils.draw_marks(s_flip_img, flip_marks, draw_idx = True)
        cv2.imshow("flip", s_flip_img)

        if utils.opencv_pause_quit_window():
            break
    return

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Pass the dataset dir as command line arg"

    img_paths_l = []
    for ext in ( '.png', '.jpg' ):
        img_paths_l.extend(glob(path.join(sys.argv[1], '*', '*'+ext)))

    show_img_with_ann(img_paths_l, n_points=6)
