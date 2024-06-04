# Facial Landmark Detection using CNN

To begin with, this project builds upon [ccn-facial-landmark](https://github.com/yinguobing/cnn-facial-landmark).

Here is a quick demo
![Demo](./demo.gif?raw=true)

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
Since this dataset is composed of AVI video files (unlike the others), I have a separate data preparation code specific to it (File `prep_300vw_dataset.py`).
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
python prep_300vw_dataset.py <dataset_loc> <tfrecord_save_loc>
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
python prep_other_datasets.py <dataset_loc> <tfrecord_save_loc>
```

## Training
Command to train the model.
```bash
python train.py github \
--tfrecords_dir <tfrecords_dir> \
--load_from <best_checkpoint_to_start_from> \ # This can be skipped
--epochs 10 --batch_size 1024 --learning_rate 0.001
```

## Eval
```bash
python train.py github \
--tfrecords_dir <tfrecords_dir> \
--load_from <model_checkpoint_to_eval> \
--eval
```

## Export
Best model can be exported to Keras native `.keras` format
```bash
python train.py github --export_model --load_from <model_to_export>
```

## Visual Test
To visually test the model on a video file or directory containing images, run the below command
```
python test.py <exported_model_file> <avi_(or)_dir_loc> (--save_video)
```
*`--save_video` will save the visual output to `output.mp4`*

## License
![License](https://img.shields.io/badge/GNU-v3.0-brightgreen)
