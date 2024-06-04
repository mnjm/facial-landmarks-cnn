from tensorflow.keras import layers, Model

# VGG-16 Like: Taken from https://github.com/yinguobing/cnn-facial-landmark
def github_model(input_shape, n_points):

    input = layers.Input(shape=input_shape, name="input")

    # Layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        padding = 'valid',
        activation='relu') (input)
    x = layers.BatchNormalization() (x)
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid') (x)

    # Layer 2
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding='valid',
        activation='relu') (x)
    x = layers.BatchNormalization() (x)
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding='valid',
        activation='relu') (x)
    x = layers.BatchNormalization() (x)
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid') (x)

    # Layer 3
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding='valid',
        activation='relu') (x)
    x = layers.BatchNormalization() (x)
    x = layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        padding='valid',
        activation='relu') (x)
    x = layers.BatchNormalization() (x)
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='valid') (x)

    # Layer 4
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3,3),
        padding='valid',
        activation='relu') (x)
    x = layers.BatchNormalization() (x)
    x = layers.Conv2D(
        filters=128,
        kernel_size=(3,3),
        padding='valid',
        activation='relu') (x)
    x = layers.BatchNormalization() (x)
    x = layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding='valid') (x)

    # Layer 5
    x = layers.Conv2D(
        filters=256,
        kernel_size=(3,3),
        padding='valid',
        activation='relu') (x)
    x = layers.BatchNormalization() (x)

    x = layers.Flatten() (x)

    # Layer 6
    x = layers.Dense(1024, activation='relu', use_bias=True) (x)
    x = layers.BatchNormalization() (x)

    # Output
    output = layers.Dense(n_points * 2, activation=None, use_bias=True, name="output") (x)

    shape_str = "x".join( [ str(x) for x in input_shape ] )
    return Model(inputs = input, outputs = output,
        name = f"github_{shape_str}_{n_points}pts")

if __name__ == "__main__":
    model = github_model((128, 128, 1), 6)
    print(model.name)
    model.summary()
