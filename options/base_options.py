import argparse
import os
import models


class Base_Options():
    """This classification defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset classification and model classification.
    """

    def __init__(self):
        """Reset the classification; indicates the classification hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--data_root', type=str, default='/home/hnu2/WLS/ContestCD/CDdata', help='path to dataroot')
        parser.add_argument('--dataset', type=str, default='contest', help='[LEVIR-CD-CD|WHU]')
        parser.add_argument('--experiment_name', type=str, default='baseline', help='name of the experiment. It decides where to load datafiles, store samples and models')
        parser.add_argument('--save_path', type=str, default='/home/hnu2/WLS/ContestCD/save', help='models are saved here')
        parser.add_argument('--data_inform_path', type=str, default='/home/hnu2/WLS/ContestCD/datafiles', help='path to files about the datafiles information')

        # model parameter
        parser.add_argument('--backbone', type=str, default='mit_b5', help='which resnet')
        parser.add_argument('--out_stride', type=int, default=32, help='out_stride')
        # train parameters
        parser.add_argument('--batch_size', type=int, default=24, help='input batch size')
        parser.add_argument('--pin', type=bool, default=True, help='pin_memory or not')
        parser.add_argument('--mixup', type=bool, default=True, help='mixup or not')

        parser.add_argument('--num_workers', type=int, default=12, help='number of workers')
        parser.add_argument('--img_size', type=int, default=512, help='image size')
        parser.add_argument('--in_channels', type=int, default=3, help='input channels')

        parser.add_argument('--num_epochs', type=int, default=125, help='num of epochs')
        parser.add_argument('--base_lr', type=float, default=1e-3, help='base learning rate')
        parser.add_argument('--decay', type=float, default=5e-4, help='decay')
        parser.add_argument('--log_interval', type=int, default=60, help='how long to log, set yo 100 batch')
        parser.add_argument('--resume', type=bool, default=False, help='resume the saved checkpoint or not')

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

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        self.opt = opt

        return self.opt


if __name__ == '__main__':
    opt = Base_Options().parse()
    print(opt)


