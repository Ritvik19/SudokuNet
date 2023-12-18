import os

os.environ["KERAS_BACKEND"] = "torch"
from keras import models, layers
from trainer import SudokuNetTrainer


class CNNTrainer(SudokuNetTrainer):
    def __init__(self, logger):
        super().__init__(logger)

    def build_model(self):
        model_input = layers.Input(shape=(9, 9, 10), name="puzzle")
        x1_1 = layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="block_01_conv_grid")(
            model_input
        )
        x1_2 = layers.Conv2D(64, kernel_size=(1, 9), padding="same", activation="relu", name="block_01_conv_row")(
            model_input
        )
        x1_3 = layers.Conv2D(64, kernel_size=(9, 1), padding="same", activation="relu", name="block_01_conv_col")(
            model_input
        )
        x = layers.Concatenate(axis=-1, name="block_01_concatenate")([x1_1, x1_2, x1_3])
        x = layers.Dropout(0.4, name="block_01_dropout")(x)
        x2_1 = layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu", name="block_02_conv_grid")(x)
        x2_2 = layers.Conv2D(64, kernel_size=(1, 9), padding="same", activation="relu", name="block_02_conv_row")(x)
        x2_3 = layers.Conv2D(64, kernel_size=(9, 1), padding="same", activation="relu", name="block_02_conv_col")(x)
        x = layers.Concatenate(axis=-1, name="block_02_concatenate")([x2_1, x2_2, x2_3])
        x = layers.Dropout(0.4, name="block_02_dropout")(x)
        x = layers.Flatten(name="flatten")(x)
        model_output = [
            layers.Dense(9, activation="softmax", name=f"position_{i+1}_{j+1}")(x) for i in range(9) for j in range(9)
        ]

        model = models.Model(
            inputs={"puzzle": model_input},
            outputs={f"position_{i+1}_{j+1}": model_output[i * 9 + j] for i in range(9) for j in range(9)},
            name="SuDoKuNetCNN",
        )
        return model
