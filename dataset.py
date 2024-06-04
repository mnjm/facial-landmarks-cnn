import tensorflow as tf
import numpy as np
import utils

def convert_to_example_and_serialize(sample_name, img, marks):
    img, marks = img.astype(np.uint8), marks.astype(np.uint8)
    def _bytes_feature(x): return tf.train.Feature(bytes_list = tf.train.BytesList(value=[x]))
    feature = {
        'name':  _bytes_feature(sample_name.encode('utf-8')),
        'image': _bytes_feature(img.tobytes()),
        'marks': _bytes_feature(marks.tobytes())
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()

class MarkDataset:
    flip_six_mark_idx = tf.constant(utils.flip_six_mark_idx)

    def __init__(self, tf_record_files_pattern, input_shape, n_points, batch_size, aug_flip_p = 0.5, aug_seed = None):
        self.tf_record_files_pattern = tf_record_files_pattern
        self.input_shape = input_shape
        self.n_points = n_points
        self.batch_size = batch_size
        self.aug_flip_p = aug_flip_p
        self.aug_seed = aug_seed

    def _parse_tfrecord(self, example):
        feature_description = {
            'name': tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
            "marks": tf.io.FixedLenFeature([], tf.string)
        }
        features = tf.io.parse_single_example(example, feature_description)
        img = tf.reshape(tf.io.decode_raw(features['image'], tf.uint8), self.input_shape)
        marks = tf.reshape(tf.io.decode_raw(features['marks'], tf.uint8), [self.n_points * 2])

        # normalize image
        img = tf.math.divide(tf.cast(img, tf.float32), tf.constant(255.0))
        marks = tf.cast(marks, tf.float32)
        return img, marks

    def _augment_random_flip(self, img, marks):
        # (1-p) do not augment data
        if tf.random.uniform([], seed = self.aug_seed) > self.aug_flip_p:
            return img, marks

        # Flip the image horizontally
        ret_img = tf.image.flip_left_right(img)

        assert self.n_points == 6, "Only supports 6 points"
        marks = tf.reshape(marks, (self.n_points, 2))

        # Flip the marks and adjust x
        ret_marks = tf.gather(marks, MarkDataset.flip_six_mark_idx)
        ret_marks = tf.stack([img.shape[1] - ret_marks[:, 0], ret_marks[:, 1]], axis=1)
        ret_marks = tf.reshape(ret_marks, [self.n_points * 2])

        return ret_img, ret_marks

    def get_dataset_pipeline(self):
        AUTO = tf.data.experimental.AUTOTUNE
        files = tf.io.matching_files(self.tf_record_files_pattern)
        files = tf.random.shuffle(files)
        shards = tf.data.Dataset.from_tensor_slices(files)
        dataset = shards.interleave(tf.data.TFRecordDataset)
        dataset = dataset.map(map_func=self._parse_tfrecord, num_parallel_calls = AUTO)
        dataset = dataset.shuffle(buffer_size = self.batch_size * 4)
        if self.aug_flip_p > 0:
            # Flip Augmentation
            dataset = dataset.map(map_func=self._augment_random_flip, num_parallel_calls = AUTO)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size = AUTO)
        return dataset

if __name__ == "__main__":

    import sys
    from utils import draw_marks, opencv_pause_quit_window
    import cv2
    from os import path

    assert len(sys.argv) == 2, "Pass the tfrecord file's dir path as command line arg"
    pattern = path.join(sys.argv[1], '*.tfrecord')
    print(f"pattern: {pattern}")

    marks_ds = MarkDataset(pattern, (128, 128, 1), 6, 1)
    ds = marks_ds.get_dataset_pipeline()

    for img, marks in ds.as_numpy_iterator():

        img = img.reshape(128,128) * 255
        img = img.astype(np.uint8)
        marks = marks.reshape(6, 2).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        draw_marks(img, marks, draw_idx = True)

        cv2.imshow("Final", img)
        if opencv_pause_quit_window(300): break
