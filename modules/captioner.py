import yaml
import os
import torch
import torch.nn as nn

from pythia.utils.configuration import ConfigNode
from pythia.tasks.processors import VocabProcessor, CaptionProcessor
from pythia.models.butd import BUTD
from pythia.common.registry import registry
from .constrained_beam_search import ConstrainedBeamSearch


config_file = 'model_data/butd.yaml'
vocab_file = 'model_data/vocabulary_captioning_thresh5.txt'
butd_checkpoint = 'model_data/butd.pth'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PythiaCaptioner(nn.Module):
    def __init__(self, use_constrained=False):
        super(PythiaCaptioner, self).__init__()
        # load configuration file
        with open(config_file) as f:
            config = yaml.load(f)
        config = ConfigNode(config)

        self.use_constrained = use_constrained

        # the following blocks of code read some configuration
        # parameter in Pythia
        config.training_parameters.evalai_inference = True
        registry.register("config", config)
        self.config = config

        captioning_config = config.task_attributes.captioning.dataset_attributes.coco
        text_processor_config = captioning_config.processors.text_processor
        caption_processor_config = captioning_config.processors.caption_processor
        # text_processor and caption_processor are used to pre-process the text
        text_processor_config.params.vocab.vocab_file = vocab_file
        caption_processor_config.params.vocab.vocab_file = vocab_file
        self.text_processor = VocabProcessor(text_processor_config.params)
        self.caption_processor = CaptionProcessor(caption_processor_config.params)

        registry.register("coco_text_processor", self.text_processor)
        registry.register("coco_caption_processor", self.caption_processor)

        self.model = self._build_model()

    def _build_model(self):
        state_dict = torch.load(butd_checkpoint)
        model_config = self.config.model_attributes.butd
        # specify the root directory of pre-trained model
        model_config.model_data_dir = os.path.dirname(os.path.abspath('__file__'))
        if self.use_constrained:
            model_config.inference.type = 'constrained_beam_search'

        model = BUTD(model_config)
        model.build()
        model.init_losses_and_metrics()

        if list(state_dict.keys())[0].startswith('module') and not hasattr(model, 'module'):
            state_dict = self._multi_gpu_state_to_single(state_dict)
          
        # load pre-trained state dict
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model

    def _multi_gpu_state_to_single(self, state_dict):
        new_sd = {}
        for k, v in state_dict.items():
            if not k.startswith('module.'):
                raise TypeError("Not a multiple GPU state of dict")
            k1 = k[7:]
            new_sd[k1] = v
        return new_sd

    def forward(self, sample_list):
        return self.model(sample_list)