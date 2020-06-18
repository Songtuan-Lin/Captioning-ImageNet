import torch
import torch.nn as nn

from pythia.common.sample import Sample, SampleList


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PythiaBUTD(nn.Module):
    def __init__(self, encoder, captioner):
        super(PythiaBUTD, self).__init__()

        self.encoder = encoder
        self.decoder = captioner

    def forward(self, images, image_scales, transitions=None):
        feature_list = self.encoder(images, image_scales)
        image_features = feature_list[0]
        assert len(feature_list) == 1, 'current model only support batch size 1'

        sample = Sample()
        sample.dataset_name = "coco"
        sample.dataset_type = "test"
        sample.image_feature_0 = image_features
        # it seems answers work as a place holder here
        # hence, it does not matter what it's size is
        sample.answers = torch.zeros((1, 10), dtype=torch.long)
        sample_list = SampleList([sample])
        sample_list = sample_list.to(device)
        # set_trace()
        if transitions is not None:
            sample_list.transitions = transitions

        output = self.decoder(sample_list)
        tokens = output['captions']
        caption = tokens.tolist()[0]
        caption = self.decoder.caption_processor(caption)['caption']

        return caption