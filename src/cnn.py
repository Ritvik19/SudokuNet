import os

os.environ["KERAS_BACKEND"] = "torch"
from keras import models, layers
from trainer import SudokuNetTrainer


class CNNTrainer(SudokuNetTrainer):
    def __init__(self, logger):
        super().__init__(logger)

    def build_model(self):
        model_input = layers.Input(shape=(9, 9, 10), name="puzzle")
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block_01_conv")(model_input)
        x = layers.Dropout(0.4, name="block_01_dropout")(x)
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="block_02_conv")(x)
        x = layers.Dropout(0.4, name="block_02_dropout")(x)
        x = layers.Flatten(name="flatten")(x)
        model_output = [
            layers.Dense(9, activation="softmax", name=f"position_{i+1}_{j+1}")(x) for i in range(9) for j in range(9)
        ]

        model = models.Model(
            inputs={"puzzle": model_input},
            outputs={f"position_{i+1}_{j+1}": model_output[i * 9 + j] for i in range(9) for j in range(9)},
            name="SuDoKuNetFFN",
        )
        return model