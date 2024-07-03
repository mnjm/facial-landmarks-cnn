from tensorflow import keras as K
import cv2
import numpy as np
from data.read_300vw import read_video_with_annot
from data.read_others import read_img_with_annot
from os import path
from argparse import ArgumentParser
from data.utils import scale_bbox_and_crop, resize_to_input_shape, apply_img_hist, draw_marks
from data.utils import opencv_pause_quit_window
from glob import glob
from headpose import draw_headpose

def get_cl_args():
    parser = ArgumentParser("Test model on dataset")
    parser.add_argument("model_file", help=".keras model file")
    parser.add_argument("loc", help="path images folder or .avi file")
    parser.add_argument("--n_points", default=6, type=int, help = "How many points?")
    parser.add_argument("--save_video", default=False, help = "save output video?", action='store_true')
    parser.add_argument("--draw_headpose", default=False, help = "Enable draw headpose", action="store_true")
    return parser.parse_args()

def eval_on_frame(model, img, bbox, marks, n_points=6):

    # Scale the image
    s_img, s_bbox, s_marks = scale_bbox_and_crop(img, bbox, marks, np.random.uniform(1, 1.2))

    # apply hist
    h_img = apply_img_hist(s_img)

    # resize imaeg
    r_img, r_marks = resize_to_input_shape(h_img, s_marks, (128, 128))

    # normalize
    n_img = r_img.astype(np.float32) / 255.0
    n_img = n_img.reshape([1, 128, 128, 1])

    # run the model
    p_marks = model.predict(n_img, verbose=0).reshape(n_points, 2)

    # debug
    # v_img = cv2.cvtColor(r_img, cv2.COLOR_GRAY2BGR)
    # draw_marks(v_img, p_marks, draw_idx=True)
    # cv2.imshow('crop', v_img)
    # cv2.waitKey(-1)

    # calc mse
    mse = np.mean( np.square ( r_marks.reshape(-1) - p_marks.reshape(-1) ))

    # scale bbox to original image
    x_s = (s_bbox[1, 0] - s_bbox[0, 0]) / 128.0
    y_s = (s_bbox[1, 1] - s_bbox[0, 1]) / 128.0
    p_marks *= np.array( [x_s, y_s ] )
    p_marks += s_bbox[0, :]

    return p_marks, mse

def data_gen(loc, n_points):

    if loc.endswith('.avi'):
        for name, img, bbox, marks in read_video_with_annot(loc, n_points):
            yield name, img, bbox, marks
    else:
        imgs_l = glob(path.join(loc, "*.png"))
        imgs_l.extend(glob(path.join(loc, "*.jpg")))
        for img_p in imgs_l:
            name, img, bbox, marks = read_img_with_annot(img_p, n_points)
            yield name, img, bbox, marks

def create_vid_writer(frame):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame.shape[1], frame.shape[0]))
    return vid_writer

def main():
    args = get_cl_args()

    model = K.models.load_model(args.model_file)
    # model.summary()

    vid_writer = None

    for _, img, bbox, marks in data_gen(args.loc, args.n_points):
        p_marks, _ = eval_on_frame(model, img, bbox, marks, args.n_points)
        draw_marks(img, p_marks, color=(0,255,0), draw_idx = False)

        frame = cv2.resize(img, (600, 800)) if not args.loc.endswith('.avi') else img

        if args.draw_headpose:
            draw_headpose(frame, p_marks)

        cv2.imshow("Predicted", frame)

        if opencv_pause_quit_window():
            break

        if args.save_video:
            if not vid_writer:
                vid_writer = create_vid_writer(frame)
            vid_writer.write(frame)

    if vid_writer:
        vid_writer.release()
    return

if __name__ == "__main__":
    main()
