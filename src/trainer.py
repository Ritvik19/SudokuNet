import os

os.environ["KERAS_BACKEND"] = "torch"
from keras import models, callbacks, utils
from datasets import Dataset
import numpy as np
from tqdm.auto import trange


class SudokuNetTrainer:
    def __init__(self, logger):
        self.logger = logger

    def build_model(self):
        return None

    def load_model(self, model_path):
        if model_path is not None:
            self.logger.info(f"Loading model from {model_path}")
            model = models.load_model(model_path)
        else:
            self.logger.info("Building model")
            model = self.build_model()
            utils.plot_model(model, to_file="../model.png", show_shapes=True, show_layer_names=True, expand_nested=True, show_layer_activations=True, rankdir="LR")
        model.summary()
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        return model

    def save_model(self, model, model_path):
        self.logger.info(f"Saving model to {model_path}")
        model.save(model_path)

    @staticmethod
    def preprocess_data_train(example, num_delete):
        generator = np.random.default_rng(19)
        for i in range(9):
            for j in range(9):
                example[f"position_{i+1}_{j+1}"] = int(example["solution"][i * 9 + j]) - 1
        if num_delete == -1:
            puzzle = np.array([int(digit) for digit in example["puzzle"]])
        else:
            puzzle = np.array([int(digit) for digit in example["solution"]])
            puzzle[generator.choice(81, num_delete, replace=False)] = 0
        puzzle = puzzle.reshape((9, 9))
        puzzle = utils.to_categorical(puzzle, num_classes=10)
        example["puzzle"] = puzzle
        return example

    @staticmethod
    def preprocess_data_eval(example):
        example["puzzle"] = np.array([int(digit) for digit in example["puzzle"]]).reshape((9, 9))
        example["solution"] = np.array([int(digit) for digit in example["solution"]]).reshape((9, 9))
        return example

    def train(self, model, train_dataset, valid_dataset, args):
        self.logger.info("Preprocessing data")
        preprocess_args = dict(
            function=lambda x: self.preprocess_data_train(x, args.num_delete),
            remove_columns=["solution", "difficulty", "missing", "set"],
        )
        preprocessed_train_dataset = SudokuDataGenerator(train_dataset.map(**preprocess_args).with_format("torch"), batch_size=args.batch_size)
        preprocessed_valid_dataset = SudokuDataGenerator(valid_dataset.map(**preprocess_args).with_format("torch"), batch_size=args.batch_size)

        if args.resume:
            _, _, last_epoch = os.path.basename(args.model_load).split(".")[0].split("_")
            last_epoch = int(last_epoch)
        else:
            last_epoch = 0

        self.logger.info("Training model")
        history = model.fit(
            preprocessed_train_dataset,
            epochs=args.epochs,
            initial_epoch=last_epoch,
            callbacks=[
                callbacks.ModelCheckpoint(
                    filepath=f"../models/{args.model_save}_{{epoch:02d}}.keras",
                    save_weights_only=False,
                    verbose=1,
                ),
                callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    verbose=1,
                    restore_best_weights=True,
                    min_delta=0.01,
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.1,
                    patience=2,
                    verbose=1,
                    min_delta=0.01,
                ),
            ],
            validation_data=preprocessed_valid_dataset,
            verbose=1,
        )
        self.save_model(model, f"../models/{args.model_save}.keras")
        return {"model": model, "history": history}

    def predict(self, model, puzzles):
        puzzles = puzzles.copy()
        for _ in trange((puzzles == 0).sum((1, 2)).max()):
            model_preds = model.predict(utils.to_categorical(puzzles, num_classes=10), verbose=0)
            preds = np.zeros((puzzles.shape[0], 81, 9))
            for i in range(9):
                for j in range(9):
                    preds[:, i * 9 + j] = model_preds[f"position_{i+1}_{j+1}"]
            probs = preds.max(2)
            values = preds.argmax(2) + 1
            zeros = (puzzles == 0).reshape((puzzles.shape[0], 81))
            for grid, prob, value, zero in zip(puzzles, probs, values, zeros):
                if any(zero):
                    where = np.where(zero)[0]
                    confidence_position = where[prob[zero].argmax()]
                    confidence_value = value[confidence_position]
                    grid.flat[confidence_position] = confidence_value
        return puzzles

    def is_valid_sudoku(self, board):
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        subgrids = [set() for _ in range(9)]

        for i in range(9):
            for j in range(9):
                num = board[i][j]
                subgrid_index = (i // 3) * 3 + j // 3
                if num in rows[i] or num in cols[j] or num in subgrids[subgrid_index]:
                    return False
                rows[i].add(num)
                cols[j].add(num)
                subgrids[subgrid_index].add(num)
        return True

    def evaluate(self, model, dataset):
        self.logger.info("Evaluating model")
        preprocess_args = dict(
            function=self.preprocess_data_eval,
            remove_columns=["difficulty", "missing", "set"],
        )
        preprocessed_dataset = dataset.map(**preprocess_args)
        puzzles = np.array(preprocessed_dataset["puzzle"])
        solutions = np.array(preprocessed_dataset["solution"])
        solutions_pred = self.predict(model, puzzles)
        deltas = (solutions != solutions_pred).sum((1, 2))
        accuracy = np.array([self.is_valid_sudoku(solution) for solution in solutions_pred]).astype(int).mean()
        mean_delta = np.round(deltas.mean(), 2)
        self.logger.info(f"[Accuracy: {accuracy}] [Mean Delta: {mean_delta}]")
        return {
            "accuracy": accuracy,
            "mean_delta": mean_delta,
        }

    def __call__(self, args):
        self.logger.info(
            f"[Training file: {[file.split('/')[-1].split('.')[0] for file in args.train]}] [Validation file: {[file.split('/')[-1].split('.')[0] for file in args.valid]}]"
        )
        self.logger.info(
            f"[Model type: {args.model_type}] [Number of digits to delete: {args.num_delete}] [Epochs: {args.epochs}] [Batch size: {args.batch_size}]"
        )
        train_dataset = Dataset.from_parquet(args.train)
        valid_dataset = Dataset.from_parquet(args.valid)
        model = self.load_model(args.model_load)
        assets = self.train(model, train_dataset, valid_dataset, args)
        assets["metrics"] = self.evaluate(assets["model"], valid_dataset)
        return assets


class SudokuDataGenerator(utils.Sequence):
    def __init__(self, dataset, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.dataset) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.dataset[idx * self.batch_size : (idx + 1) * self.batch_size]
        return {"puzzle": np.array(batch["puzzle"])}, {
            f"position_{i+1}_{j+1}": np.array(batch[f"position_{i+1}_{j+1}"]) for i in range(9) for j in range(9)
        }
