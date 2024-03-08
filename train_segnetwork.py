import os

from Trainer.ReplayTrainer import ReplayTrainer
from options.base_options import BaseOptions


def run(opt):
    if opt.method == "replay":
        Trainer = ReplayTrainer(opt)
        Trainer.Train()


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # Training settings

    opt = BaseOptions().parse()
    run(opt)
