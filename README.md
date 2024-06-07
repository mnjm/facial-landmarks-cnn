# Facial Landmark Detection using CNN


Here is a quick demo

**8 Points with Face alignment / Head Pose**

![Demo with hs](./demos/demo_8pts_hs.gif?raw=true)

**6 Points**

![Demo](./demos/demo_6pts.gif?raw=true)

**Prerequisites**

![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.16-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.26-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-blue)

## Datasets

- [300-VW](https://ibug.doc.ic.ac.uk/resources/300-VW/)
- [300-W](https://ibug.doc.ic.ac.uk/resources/300-W/)
- [AFW](https://www.ics.uci.edu/~xzhu/face/)
- [HELEN](http://www.ifp.illinois.edu/~vuongle2/helen/)
- [IBUG](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
- [LFPW](https://neerajkumar.org/databases/lfpw/)

- *Some* datasets mentioned above came pre-split into training and test sets. Others I manually split them randomly.
- Prepared dataset is stored and served to the model as tfrecord files in sharded fashion

### Datasets file structure

```
 { dataset }
 ├── testset
 │   ├── {img}.(jpg|png)
 │   ├── {img}.pts
 └── trainset
     ├── {img}.(jpg|png)
     ├── {img}.pts
```

**[300-VW](https://ibug.doc.ic.ac.uk/resources/300-VW/)**

Since this dataset is composed of (.avi) video files (unlike the others), It should be served in this format.

```
300VW
├── testset
│   ├── {sample name}
│   │   ├── annot / {frame_no}.pts
│   │   └── vid.avi
└── trainset
    ├── {sample name}
    │   ├── annot / {frame_no}.pts
    │   └── vid.avi
frame_no format "%06d"
```

Note: Dataset base dir name `300VW`, this is hardcoded in `prep_tfrecords.py`.

## Generate TFRecord files

Provide the dataset directory and the path to store the TFRecord files

```bash
python prep_tfrecords.py <dataset_loc> <tfrecord_save_loc> (--test_set) --n_points <6 (or) 8>
```

## Training

Command to train the model.

```bash
python train.py <model_type> \
 --n_points <6 (or) 8> \
 --tfrecords_dir <tfrecords_dir> \
 --load_from <best_checkpoint_to_start_from> \ # This can be skipped
 --epochs 10 --batch_size 1024 --learning_rate 0.001
```
- Check [here](https://github.com/mnjm/facial-landmarks-cnn/blob/main/train.py#L10C1-L10C57) for model types

## Evaluate

```bash
python train.py <model_type> \
 --n_points <6 (or) 8> \
 --tfrecords_dir <tfrecords_dir> \
 --load_from <model_checkpoint_to_eval> \
 --eval_model
```

## Export

Best model can be exported to Keras native `.keras` format
```bash
python train.py <model_type> \
 --load_from <checkpoint_to_export> \
 --export_model <export_as>
```

## Visual Test

To visually test the model on a video file or directory containing images, run the below command
```
python visual_test.py <exported_model_file> <avi_(or)_dir_loc> (--save_video) --n_points <6 (or) 8>
```
*`--save_video` will save the visual output to `output.mp4`*

### Visual Test with Face alignment / Head Pose

**Only works on 8pts**

To visually test the model on a video file or directory containing images, run the below command
```
python visual_test.py <exported_model_file> <avi_(or)_dir_loc> (--save_video) --n_points 8 --draw_headpose
```

## Addendum

- Model provided [here](https://github.com/mnjm/facial-landmarks-cnn/blob/main/models/from_github.py) is from [ccn-facial-landmark](https://github.com/yinguobing/cnn-facial-landmark).

## License
![License](https://img.shields.io/badge/GNU-v3.0-brightgreen)
