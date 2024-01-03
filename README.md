# SudokuNet

Ai Sudoku Solver

This repository contains scripts and resources for training Sudoku Net

## Dataset

The model is trained on a dataset comprising 17 million Sudoku puzzles from various sources ([Dataset Card](https://huggingface.co/datasets/Ritvik19/Sudoku-Dataset)). The dataset includes puzzle configurations, solutions, difficulty levels, and sources.

## Model

### Model Architecture

The solver utilizes a Feed Forward Neural Network architecture. Details about the layers, units, and unique aspects of the architecture can be found in the [Model Card](https://huggingface.co/Ritvik19/SudokuNet).

### Training

The training process involves using the Adam optimizer with a learning rate of 1e-3 and a batch size of 64K. Training and evaluation scripts are available in the `src` directory.

### Performance Metrics

The model's performance is evaluated based on accuracy, indicating whether the model correctly solves Sudoku puzzles or not. Further insights on performance metrics are detailed in the [Model Card](https://huggingface.co/Ritvik19/SudokuNet).

### Getting the Pretrained Models

To fetch pretrained models from the remote model repository

```python
from huggingface_hub import hf_hub_download

model_file_path = hf_hub_download(
    repo_id="Ritvik19/SudokuNet",
    filename="model_filename_here",
    revision="model_revision_here",
)
```

#### Pretrained Models

| Model | File Name | Revision                                 |
| :---- | :-------- | :--------------------------------------- |
| v1.0  | ffn.keras | b57f9a0538e28249c92733cb025c87d07831baa1 |

## Usage

### Interacting with the Model

Users can interact with the trained model through this [Space](https://huggingface.co/spaces/Ritvik19/SudokuNetDemo).

## Getting Started

### Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### Model Definitions

Two predefined models are available:

1. **Feed Forward Neural Network (FFN):** Defined in `ffn.py`
2. **Convolutional Neural Network (CNN):** Defined in `cnn.py`

### Custom Model Architecture

If you want to define a custom architecture, follow these steps:

1. Inherit from the `SudokuNetTrainer` class.
2. Override the `build_model` method to define your custom architecture.

Example of custom model definition:

```python
from trainer import SudokuNetTrainer

class CustomSudokuNetTrainer(SudokuNetTrainer):
    def build_model(self):
        # Define your custom architecture here
        # ...

# Use the CustomSudokuNetTrainer in main.py
```

### Training the Model

To train the model, use the `main.py` script with the following options:

```bash
usage: main.py [-h] [--train TRAIN [TRAIN ...]] [--valid VALID [VALID ...]] [--model-load MODEL_LOAD] [--model-save MODEL_SAVE] [--model-type MODEL_TYPE]
               [--num-delete NUM_DELETE] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--resume]

train sudoku net

options:
  -h, --help            show this help message and exit
  --train TRAIN [TRAIN ...]
                        the path of the dataset file
  --valid VALID [VALID ...]
                        the path of the dataset file
  --model-load MODEL_LOAD
                        the path of the model file to load
  --model-save MODEL_SAVE
                        the path of the model file to save
  --model-type MODEL_TYPE
                        the type of model to train
  --num-delete NUM_DELETE
                        the number of digits to delete
  --epochs EPOCHS       the number of epochs to train
  --batch-size BATCH_SIZE
                        the batch size to train
  --resume              resume training from the model file
```

Adjust the options according to your requirements. For example, to train the FFN model with a specific dataset, use:

```bash
python main.py --train path/to/train_data.parquet --valid path/to/valid_data.parquet --model-type ffn
```

Feel free to customize the training parameters as needed.

### Inference

Performing inference with the trained model can be done using the `SudokuSolver` class:

1. **Instantiate the SudokuSolver Class:**

```python
from infer import SudokuSolver

# Provide the path to the trained model
model_path = 'path/to/your/model_file'

# Instantiate the SudokuSolver object
solver = SudokuSolver(model_path)
```

2. **Solve Sudoku Puzzles:**

```python
# Provide the puzzle as input to the solver object
puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

# Call the solver object with the puzzle to get the solution
solution = solver(puzzle)
```

Replace `'path/to/your/model_file'` with the actual path to your trained model file. Input your Sudoku puzzle as a 9x9 matrix with 0s indicating empty cells. The solver object will return the solution for the provided puzzle.

## Contribution

Contributions are welcome! If you want to contribute to this project, please follow the follwing guidelines:

1. **Fork** the repository and create your branch from `main`.
2. **Discuss** major changes or enhancements by opening an issue first.
3. **Commit** changes with descriptive commit messages.
4. **Testing** is appreciated; ensure your code is thoroughly tested.
5. **Pull Requests** should be linked to an open issue and provide a clear explanation of changes.

### Code Style

- Follow consistent coding styles as present in the repository.
- Comment your code where necessary to enhance readability.

### Reporting Issues

- If you encounter bugs or have suggestions, please open an issue.
- Clearly explain the problem with steps to reproduce for bug reports.

### Feature Requests

- Open an issue to propose new features or improvements.
- Describe the feature and its potential impact.

### Pull Requests

- Link your pull request to the related issue for easy tracking.
- Provide a concise summary of changes in the PR description.

Your contributions will be highly appreciated and acknowledged!

## License

This project is licensed under Apache-2.0. Refer to `LICENSE` for more details.
