import cv2
import numpy as np

marks_map_68 = {
    # left eye, right eye and mouth points each with (left, right) pairs
    6 : [36, 39, 42, 45, 48, 54],
    # left eye, right eye, mouth points each with (left, right) pairs, nose tip and chin
    8 : [36, 39, 42, 45, 48, 54, 33, 8],
    # default map
    68 : list(range(68))
}

# map for flip augmentation
flip_map = {
    # right eye, left eye and mouth with (right followed by left pairs)
    6 : [3, 2, 1, 0, 5, 4],
    # right eye, left eye and mouth with (right followed by left pairs), nose tip and chin
    8 : [3, 2, 1, 0, 5, 4, 6, 7]
}

# read marks from .pts file
def read_pts_file(f_path):
    """
    Reads .pts file of given video and frame
    Args:
        f_path: pts file path
    Return:
        numpy array of size (68x2)
    """
    marks = np.ones((68, 2), dtype=np.float32) * np.nan
    i = 0
    with open(f_path) as f:
        for line in f:
            if "version" in line or "n_points" in line or "{" in line or "}" in line: continue
            x, y = ( float(val) for val in line.split(" ")[:2] )
            marks[i, 0], marks[i, 1] = x - 1, y - 1
            i += 1
        assert i == 68, "Failed loading 68 landmarks"
    return marks

def opencv_pause_quit_window(delay=1):
    SPACE_KEY = 32
    key = cv2.waitKey(delay)
    if key == ord('q'): return True
    elif key == SPACE_KEY:
        while cv2.waitKey(-1) != SPACE_KEY: continue
    return False

# Draws marks on images
def draw_marks(img, marks, color=(0,0,255), draw_idx=False, idx_color=(0, 255, 128)):
    for i, pt in enumerate(marks):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x,y), 2, color, -1)
        if draw_idx: cv2.putText(img, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, idx_color, 1, -1)

def draw_bbox(img, bbox, color=(255,0,0)):
    p1 = int(bbox[0,0]), int(bbox[0,1])
    p2 = int(bbox[1,0]), int(bbox[1,1])
    cv2.rectangle(img, p1, p2, color, 1)

def scale_bbox(bbox, scale):
    center = np.mean(bbox, axis=0)
    ret = center + (bbox - center) * scale
    return ret

def scale_bbox_and_crop(img, bbox, marks, scale):
    ret_bbox = scale_bbox(bbox, scale)
    ret_bbox = np.round(ret_bbox).astype(np.int32)
    (x0, y0, x1, y1) = ret_bbox.reshape(4)
    if x0 < 0: x0 = 0
    if y0 < 0: y0 = 0
    if x1 > img.shape[1]: x1 = img.shape[1]
    if y1 > img.shape[0]: y1 = img.shape[0]
    ret_bbox = np.array( [ [x0, y0], [x1, y1] ] )
    ret_img = img[y0:y1, x0:x1]
    ret_marks = marks - ret_bbox[0, :]
    return ret_img, ret_bbox, ret_marks

def resize_to_input_shape(img, marks, input_shape):
    ret_img = cv2.resize(img, input_shape)
    ret_marks = np.copy(marks)
    s_w = float(input_shape[1]) / img.shape[1]
    s_h = float(input_shape[0]) / img.shape[0]
    ret_marks[:, 0] *= s_w
    ret_marks[:, 1] *= s_h
    return ret_img, ret_marks

def apply_img_hist(img):
    return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

def augment_flip(img, marks):
    # flip image
    ret_img = cv2.flip(img, 1)

    # map points
    n_points = marks.shape[0]
    assert n_points in flip_map.keys(), f"Flip map for {n_points}pts not found"
    ret_marks = marks[flip_map[n_points], :]

    # offset
    ret_marks[:, 0] = img.shape[1] - ret_marks[:, 0]
    return ret_img, ret_marks

def bbox_from_marks(marks):
    min_x, min_y = np.min(marks[:, 0]), np.min(marks[:, 1])
    max_x, max_y = np.max(marks[:, 0]), np.max(marks[:, 1])
    return np.array([ [min_x, min_y], [max_x, max_y] ])
