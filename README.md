# Facial Landmark Detection using CNN

To begin with, this project builds upon [ccn-facial-landmark](https://github.com/yinguobing/cnn-facial-landmark).

Here is a quick demo

**8 Points with Face alignment**

![Demo with hs](./demos/demo_8pts_hs.gif?raw=true)

**6 Points**

![Demo](./demo_6pts.gif?raw=true)

**Dependencies**

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

### Dataset Preparation
- *Some* datasets mentioned above came pre-split into training and test sets. Others I manually split them randomly.
- Prepared dataset is stored and served to the model as tfrecord files in sharded fashion

### [300-VW](https://ibug.doc.ic.ac.uk/resources/300-VW/)
Since this dataset is composed of AVI video files (unlike the others), I have a separate data preparation code specific to it (File `prep_300vw.py`).
**Dataset file structure**
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
To run, pass the dataset dir and path to store the tf record file,
```bash
python prep_300vw.py <dataset_loc> <tfrecord_save_loc> (--test_set) --n_points <6 (or) 8>
```

## Other datasets
**Dataset file structure**
```
 { dataset }
 ├── testset
 │   ├── {img}.(jpg|png)
 │   ├── {img}.pts
 └── trainset
     ├── {img}.(jpg|png)
     ├── {img}.pts
```
To run, provide the dataset directory and the path to store the TFRecord file
```bash
python prep_others.py <dataset_loc> <tfrecord_save_loc> (--test_set) --n_points <6 (or) 8>
```

## Training
Command to train the model.
```bash
python train.py <model_type> \
--tfrecords_dir <tfrecords_dir> \
--load_from <best_checkpoint_to_start_from> \ # This can be skipped
--epochs 10 --batch_size 1024 --learning_rate 0.001
```
- Check [here](https://github.com/mnjm/facial-landmarks-cnn/blob/main/train.py#L10C1-L10C57) for model types

## Eval
```bash
python train.py <model_type> \
--tfrecords_dir <tfrecords_dir> \
--load_from <model_checkpoint_to_eval> \
--eval
```

## Export
Best model can be exported to Keras native `.keras` format
```bash
python train.py <model_type> --export_model --load_from <model_to_export> --_n_points <6 (or) 8>
```

## Visual Test
To visually test the model on a video file or directory containing images, run the below command
```
python visual_test.py <exported_model_file> <avi_(or)_dir_loc> (--save_video) --n_points <6 (or) 8>
```
*`--save_video` will save the visual output to `output.mp4`*

## Visual Test with Face alignment

**Only works on 8pts**
To visually test the model on a video file or directory containing images, run the below command
```
python visual_test_headpose.py <exported_model_file> <avi_(or)_dir_loc> (--save_video) --n_points 8
```
*`--save_video` will save the visual output to `output.mp4`*

## License
![License](https://img.shields.io/badge/GNU-v3.0-brightgreen)
