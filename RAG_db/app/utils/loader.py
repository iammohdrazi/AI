import time


class Loader:
    def __init__(self, logger):
        self.logger = logger

    def start(self, message: str):
        self.logger.info(message)

    def step(self, message: str):
        self.logger.info(message)

    def done(self, message: str):
        self.logger.info(message)
