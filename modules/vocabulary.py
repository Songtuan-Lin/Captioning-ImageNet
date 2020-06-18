import yaml

from pythia.utils.configuration import ConfigNode
from pythia.tasks.processors import VocabProcessor, CaptionProcessor, GloVeProcessor

class Vocabulary:
    def __init__(self):
        config_file = 'model_data/butd.yaml'
        vocab_file = 'model_data/vocabulary_captioning_thresh5.txt'

        with open(config_file) as f:
            config = yaml.load(f)
        config = ConfigNode(config)

        captioning_config = config.task_attributes.captioning.dataset_attributes.coco
        text_processor_config = captioning_config.processors.text_processor
        text_processor_config.params.vocab.vocab_file = vocab_file
        text_processor = VocabProcessor(text_processor_config.params)

        self.vocab = text_processor.vocab

    def get_vocabulary(self):
        return self.vocab

if __name__ == "__main__":
    Vocabulary()  # download vocabulary