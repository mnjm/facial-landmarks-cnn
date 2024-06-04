from tensorflow import keras as K
from os import path, mkdir
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataset import MarkDataset
from models.from_github import github_model
from models.vgg16 import vgg16
from math import ceil
from datetime import datetime

which_model = { 'github': github_model, 'vgg16': vgg16 }
INPUT_SHAPE = (128, 128, 1)
N_POINTS = 6

N_TRAINING_SAMPLES = 188772
N_VAL_SAMPLES = 34262

def get_cl_args():
    parser = ArgumentParser(description="Training", formatter_class = ArgumentDefaultsHelpFormatter)
    parser.add_argument("model_type", help="Avai: " + ",".join(which_model.keys()), type = str)
    parser.add_argument("--tfrecords_dir", default=None, type = str,
                        help="Tfrecords dir: will lookfor files with `trainset` and `testset` in it name")
    parser.add_argument("--eval_model", default=False, action = 'store_true',
                        help="Eval the model, do not train")
    parser.add_argument("--export_model", default=False, action = 'store_true',
                        help="Export model: remove optimizers and export the file")
    parser.add_argument("--epochs", default=1, help="Epochs to run", type = int)
    parser.add_argument("--batch_size", default=64, help="Batch_size", type = int)
    parser.add_argument("--learning_rate", default=0.01, help="learning_rate", type = float)
    parser.add_argument("--load_from", default=None, help="Load weights from?", type = str)
    parser.add_argument("--save_dir", default="./", help="Dir to store checkpoints, logs and exported model")
    return parser.parse_args()

def get_dataset(args, val_set = False):
    train_or_test = "testset" if val_set else "trainset"
    assert path.isdir(args.tfrecords_dir), "Pass the tfrecords dir"
    ds = MarkDataset(
            path.join(args.tfrecords_dir, f"*_{train_or_test}_*.tfrecord"),
            INPUT_SHAPE,
            N_POINTS,
            args.batch_size,
            aug_flip_p = 0.5,
            aug_seed = 123
    )
    return ds.get_dataset_pipeline()

def main():
    args = get_cl_args()

    # Load model
    model = which_model[args.model_type](INPUT_SHAPE, N_POINTS)
    print(f"Model name: {model.name}")
    print("="*100)
    model.summary()
    if args.load_from:
        model.load_weights(args.load_from)

    if args.export_model:
        assert args.load_from, "Pass the model to export"

        export_to = path.join(args.save_dir, model.name + "_exported.keras")
        assert not path.isfile(export_to), f"{export_to} exists!, move it somewhere and rerun exporting"

        model.save(export_to)
        print(f"Model saved at {export_to}")
        return

    # Compike
    model.compile(
            optimizer = K.optimizers.Adam(learning_rate = args.learning_rate),
            loss = K.losses.mean_squared_error
    )

    assert args.tfrecords_dir, "Pass the tfrecords pls"

    ch_path = path.join(args.save_dir, "checkpoints")
    if not path.isdir(ch_path): mkdir(ch_path)
    logs_path = path.join(args.save_dir, "logs")
    if not path.isdir(ch_path): mkdir(logs_path)


    # Dataset
    train_ds, val_ds = get_dataset(args), get_dataset(args, val_set = True)
    train_steps = ceil(float(N_TRAINING_SAMPLES) / args.batch_size)
    val_steps = ceil(float(N_VAL_SAMPLES) / args.batch_size)

    if args.eval_model:
        assert path.isfile(args.load_from), "Pass the model checkpoint to evalulate"
        model.evaluate(val_ds, steps = val_steps, verbose = 1)
        return

    # logs and checkpoint
    time_str = datetime.now().strftime("%d_%m_%y_%H%M")
    ck_pt = K.callbacks.ModelCheckpoint(
            filepath = path.join(ch_path,
                                 f"{model.name}_{time_str}_lr{args.learning_rate}_"  + "{epoch:02d}-{val_loss:.2f}.keras"),
            save_weights_only = False,
            save_best_only = True,
            monitor = 'val_loss',
            verbose = 1
        )
    tboard = K.callbacks.TensorBoard(log_dir = logs_path)
    cbs = [ tboard, ck_pt ]


    # Training time
    model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = args.epochs,
        steps_per_epoch = train_steps,
        validation_steps = val_steps,
        callbacks = cbs
    )

if __name__ == "__main__": main()
