import argparse
from notify import SandeshLogger
from ffn import FFNTrainer
from cnn import CNNTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train sudoku net")
    parser.add_argument("--train", type=str, help="the path of the dataset file", nargs="+")
    parser.add_argument("--valid", type=str, help="the path of the dataset file", nargs="+")
    parser.add_argument("--model-load", type=str, help="the path of the model file to load")
    parser.add_argument("--model-save", type=str, help="the path of the model file to save")
    parser.add_argument("--model-type", type=str, default="ffn", help="the type of model to train")
    parser.add_argument("--num-delete", type=int, default=-1, help="the number of digits to delete")
    parser.add_argument("--epochs", type=int, default=100, help="the number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=65536, help="the batch size to train")
    parser.add_argument("--resume", action="store_true", help="resume training from the model file")
    args = parser.parse_args()

    logger = SandeshLogger("SuDoKuNet")
    result_logger = SandeshLogger("result", log_file="../result.log")

    logger.info("Training started")
    logger.send_sandesh(
        f"Training started. [model={args.model_type} num_delete={args.num_delete}, epochs={args.epochs}]"
    )
    if args.model_type == "ffn":
        trainer = FFNTrainer(logger)
    elif args.model_type == "cnn":
        trainer = CNNTrainer(logger)
    else:
        raise ValueError(f"Unknown model type {args.model_type}")
    assets = trainer(args)
    logger.send_sandesh(
        f"Training completed. [{args.model_save}] [accuracy={assets['metrics']['accuracy']}] [mean_delta={assets['metrics']['mean_delta']}]"
    )
    result_logger.info(
        f"[model={args.model_type}] [num_delete={args.num_delete}] "
        + f"[accuracy={assets['metrics']['accuracy']}] [mean_delta={assets['metrics']['mean_delta']}]"
    )
    logger.info("Training completed.")
