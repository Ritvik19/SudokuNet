import sandesh
import argparse
import logging


class SandeshLogger:
    def __init__(self, name, disable=False, log_file="../training.log"):
        self.name = name
        self.logger = self.setup_logger(name, log_file)
        self.webhook = open("../../webhook.txt", "r").read().strip()
        self.disable = disable

    def setup_logger(self, name, log_file="../training.log"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger

    def send_sandesh(self, message):
        try:
            if not self.disable:
                sandesh.send(f"{self.name}: {message}", webhook=self.webhook)
        except Exception:
            pass

    def info(self, message, sandesh=False):
        if not self.disable:
            self.logger.info(message)
            if sandesh:
                self.send_sandesh(message)

    def debug(self, message, sandesh=False):
        if not self.disable:
            self.logger.debug(message)
            if sandesh:
                self.send_sandesh(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="Training completed.")
    args = parser.parse_args()
    logger = SandeshLogger("main")
    logger.send_sandesh(args.message)
