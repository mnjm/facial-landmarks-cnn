import cv2
import numpy as np

marks_3d = np.array([
                        (-225.0, 170.0, -135.0),     # Left eye left corner
                        (225.0, 170.0, -135.0),      # Right eye right corner
                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                        (150.0, -150.0, -125.0),     # Right mouth corner
                        (0.0, 0.0, 0.0),             # Nose tip
                        (0.0, -330.0, -65.0)         # Chin
                        ])
# Corresponding 2D landmarks
marks_3d_to_8pts_2d_map = np.array((0, 3, 4, 5, 6, 7), dtype=np.uint32)

def get_camera_matrix(imgSize):
    focal_l = imgSize[1]
    center = (imgSize[1]/2, imgSize[0]/2)
    cam_mat = np.array([
                        (focal_l, 0, center[0]),
                        (0, focal_l, center[1]),
                        (0, 0, 1)
                        ], dtype=np.float64)
    return cam_mat

def draw_headpose(img, marks):
    assert marks.shape[0] == 8, "Headpose is only supported on 8pt marks"
    marks = marks[marks_3d_to_8pts_2d_map, :].astype(np.float64)
    camera_mat = get_camera_matrix(img.shape)
    dist_coeffs = np.zeros((4, 1))
    success, r_vec, t_vec = cv2.solvePnP(marks_3d, marks, camera_mat,
                                            dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return

    nose_vec_projected = cv2.projectPoints(np.array([(0,0,250.0)]), r_vec, t_vec,
                                            camera_mat, dist_coeffs)[0]

    point1 = (int(marks[-2,0]), int(marks[-2,1]))
    point2 = (int(nose_vec_projected[0][0][0]), int(nose_vec_projected[0][0][1]))
    cv2.line(img, point1, point2, (255,0,0), 2)
    return
