import argparse
import Trainer


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""

        # Training settings
        parser = argparse.ArgumentParser(description='Continual learning')
        parser.add_argument('--framwork_path', type=str, default='CGR', metavar='N',
                            help='')
        parser.add_argument('--exp_name', type=str, default='replay', metavar='N',
                            help='Name of the experiment')
        parser.add_argument('--mode', type=str, default='class_incremental', metavar='N',
                            help='Choose continual learning mode, see tasks.py')
        parser.add_argument('--method', type=str, default='replay', metavar='N',
                            help='Name of method, eg: replay')
        parser.add_argument('--model', type=str, default='res_unet', metavar='N',
                            help='Name of model, eg: res_unet')
        parser.add_argument('--dataroot', type=str, default="MultiTasksDataset", metavar='N',
                            help='dataset path')
        parser.add_argument('--seed', type=int, default=12345, metavar='N',
                            help='Choose random seed')
        parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                            help='Size of train batch')
        parser.add_argument('--val_batch_size', type=int, default=1, metavar='N',
                            help='Size of test batch')
        parser.add_argument('--start_iteration', type=int, default=0, metavar='N',
                            help='number of episode to train ')
        parser.add_argument('--total_iterations', type=int, default=40000, metavar='N',
                            help='number of episode to train ')
        parser.add_argument('--lr', type=float, default=2e-4, metavar='N',
                            help='learning rate')
        parser.add_argument('--device', type=str, default='cuda', metavar='N',
                            help='gpu for training')
        parser.add_argument('--dataset_json', type=str, default="Dataset_fingerprint",
                            metavar='N',
                            help='base dataset fingerprint json fold')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()

        if opt.method == 'replay':
            model_option_setter = Trainer.ReplayTrainer.ReplayTrainer.modify_commandline_options
            parser = model_option_setter(parser)
            opt, _ = parser.parse_known_args()
        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)

        self.opt = opt
        return self.opt
