import tensorflow as tf
import numpy as np
from data import convert_to_example_and_serialize
from argparse import ArgumentParser
from glob import glob
from os import path
from data.read_300vw import Video_300VW
from data.utils import scale_bbox_and_crop, resize_to_input_shape, apply_img_hist, marks_map_68

SHARD_SIZE = 1024
INPUT_SHAPE = (128, 128, 1)

def prep_args():
    parser = ArgumentParser()
    parser.add_argument('dataset', help="Dataset directory")
    parser.add_argument('outdir', help="Output dir")
    parser.add_argument('--test_set', help="Prepare test set", action='store_true')
    parser.add_argument('--n_points', help="Number of points", default=6, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = prep_args()

    assert args.n_points in marks_map_68.keys(), f"Invalid n_points it should be " + ",".join( str(x) for x in marks_map_68.keys() )
    subdir = "testset" if args.test_set else "trainset"

    vid_files = glob(path.join(args.dataset, subdir, '*', 'vid.avi'))
    print(f"n_vid_file = {len(vid_files)}")

    count, shard_idx, shard_sample_count = 0, 0, 0
    tf_record_base_name = path.join(args.outdir, "300vw_" + subdir + "_" + \
                                    "x".join([str(x) for x in INPUT_SHAPE]) + f"_{args.n_points}p_")

    # create first tf record file
    writer = tf.io.TFRecordWriter(tf_record_base_name + f"{shard_idx}.tfrecord")

    vid_objs = [ Video_300VW(x, args.n_points) for x in vid_files ]

    done_flag = True
    while done_flag:

        done_flag = False
        for vid_idx, vidobj in enumerate(vid_objs):

            got_frame, s_name, img, bbox, marks = vidobj.next_frame()
            done_flag = done_flag or got_frame
            if not got_frame:
                continue

            print(f"{count} Adding sample: {s_name}")
            count += 1

            # scale and crop img
            img, _, marks = scale_bbox_and_crop(img, bbox, marks, np.random.uniform(1.1, 1.3))
            assert INPUT_SHAPE[2] == 1, "For hist eq preproc, input should have 1 channel (gray)"
            # apply histogram equalization
            img = apply_img_hist(img)
            # resize img and scale landmarks accordingly
            img, marks = resize_to_input_shape(img, marks, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
            assert np.all(marks >= 0) and np.all(marks <= INPUT_SHAPE[0]), f"Marks out of bounds {s_name}"

            writer.write(convert_to_example_and_serialize(s_name, img, marks))
            shard_sample_count += 1
            if shard_sample_count >= SHARD_SIZE:
                writer.close()
                shard_idx += 1
                shard_sample_count = 0
                writer = tf.io.TFRecordWriter(tf_record_base_name + f"{shard_idx}.tfrecord")
