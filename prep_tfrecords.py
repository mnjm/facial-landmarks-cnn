import tensorflow as tf
import numpy as np
from data import convert_to_example_and_serialize
from argparse import ArgumentParser
from glob import glob
from os import path
from data.read_300vw import Video_300VW
from data.read_others import read_img_with_annot
from data.utils import scale_bbox_and_crop, resize_to_input_shape, apply_img_hist, marks_map_68

SHARD_SIZE = 1024
INPUT_SHAPE = (128, 128, 1)

def prep_args():
    parser = ArgumentParser("TFRec prepare script")
    parser.add_argument('dataset', help="Dataset directory")
    parser.add_argument('outdir', help="Output dir")
    parser.add_argument('--test_set', help="Prepare test set", action='store_true')
    parser.add_argument('--n_points', help="Number of points", default=6, type=int)
    return parser.parse_args()

def ds_gen(dataset_path, n_points):
    # init
    imgs_path = []
    for ext in ( '.png', '.jpg' ):
        imgs_path.extend(glob(path.join(dataset_path, '*' + ext)))

    print(f"No. of img files found: {len(imgs_path)}")

    # iternate
    for img_path in imgs_path:
        s_name, img, bbox, marks = read_img_with_annot(img_path, n_points)
        yield s_name, img, bbox, marks
    return

def ds_gen_300vw(dataset_path, n_points):
    # init
    vid_files = glob(path.join(dataset_path, '*', 'vid.avi'))
    vid_objs = [ Video_300VW(x, n_points) for x in vid_files ]
    print(f"No. of video files found: {len(vid_objs)}")

    not_done = True
    # iterleave iternate
    while not_done:
        not_done = False
        for vidobj in vid_objs:
            got_frame, s_name, img, bbox, marks = vidobj.next_frame()
            not_done = not_done or got_frame
            if not got_frame:
                continue
            yield s_name, img, bbox, marks
    return

def preprocess(img, bbox, marks):
    # scale and crop img
    img, _, marks = scale_bbox_and_crop(img, bbox, marks, np.random.uniform(1.0, 1.2))
    # apply histogram equalization
    img = apply_img_hist(img)
    # resize img and scale landmarks accordingly
    img, marks = resize_to_input_shape(img, marks, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    return img, marks

if __name__ == "__main__":
    args = prep_args()

    assert args.n_points in marks_map_68.keys(), f"Invalid n_points it should be " + ",".join( str(x) for x in marks_map_68.keys() )

    dataset_name = path.basename(args.dataset)
    subdir = "testset" if args.test_set else "trainset"

    if dataset_name == "300VW":
        print("#"*20)
        print("Detected as 300VW dataset")
        print("#"*20)
        ds_iter_gen = ds_gen_300vw
    else:
        ds_iter_gen = ds_gen

    dataset_path = path.join(args.dataset, subdir)
    assert path.isdir(dataset_path), f"Failed to read {dataset_path} as a dir"

    # create first tf record file
    tf_record_base_name = path.join(args.outdir, f"{dataset_name}_{subdir}_" + \
                                    "x".join([str(x) for x in INPUT_SHAPE]) + f"_{args.n_points}pts_")
    count, shard_idx, shard_sample_count = 0, 0, 0
    create_new_tfrec = lambda: tf.io.TFRecordWriter(tf_record_base_name + f"{shard_idx}.tfrecord")
    writer = create_new_tfrec()

    for s_name, img, bbox, marks in ds_iter_gen(dataset_path, args.n_points):
        print(f"{count} Adding sample: {s_name}")
        count += 1

        img, marks = preprocess(img, bbox, marks)

        writer.write(convert_to_example_and_serialize(s_name, img, marks))
        shard_sample_count += 1
        if shard_sample_count >= SHARD_SIZE:
            writer.close()
            shard_idx += 1
            shard_sample_count = 0
            writer = create_new_tfrec()
