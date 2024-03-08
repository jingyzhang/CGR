import os
from utils.utils import IOStream, mkdir, load_json
from abc import ABC, abstractmethod
import torch
from monai.transforms import AsDiscrete
from utils import tasks

class BaseTrainer(ABC):
    def __init__(self, opt):
        self.opt = opt
        self.exp_name = opt.exp_name
        self.model = opt.model
        self.method = opt.method
        self.dataset_json = []
        self.device = []
        self.base_checkpoint_path = os.path.join(opt.framwork_path, "experiment", self.exp_name)
        mkdir(self.base_checkpoint_path)
        
        os.system('cp options/base_options.py ' + self.base_checkpoint_path + '/' + 'base_options.py.backup')
        if opt.exp_name.split('_')[0] == "CFP" or "PFC":
            prostate_json = os.path.join(opt.dataset_json,  "Prostate_fingerprint.json")
            fundus_json = os.path.join(opt.dataset_json,  "Fundus_fingerprint.json")
            MNMS_json = os.path.join(opt.dataset_json,  "MNMS_fingerprint.json")
            self.dataset_json = {"Prostate": load_json(prostate_json), "Fundus": load_json(fundus_json), "mnms": load_json(MNMS_json)}
        else:
            json_path = os.path.join(opt.dataset_json, opt.exp_name.split('_')[0] + "_fingerprint.json")
            self.dataset_json = load_json(json_path)
        
        self.sites_name = tasks.get_task_labels(dataset=self.opt.exp_name.split("_")[0], mode="tasks_name")
        
        self.num_classes = 1
        for name, datasetjson in self.dataset_json.items():
            self.num_classes += datasetjson['num_class'] - 1
            

    def init_io(self, log_path):
        mkdir(log_path)
        io = IOStream(log_path + "/running.log")
        return io


    def set_device(self, ):
        device = torch.device(self.opt.device)
        return device

    def get_post_process(self, num_class):
        val_post_process = AsDiscrete(argmax=True, to_onehot=num_class)
        return val_post_process

    @abstractmethod
    def Train(self,):
        pass

    @staticmethod
    def modify_commandline_options(parser):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser




