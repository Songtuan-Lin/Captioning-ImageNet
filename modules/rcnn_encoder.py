import torch
import torch.nn as nn

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.layers import nms


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

maskrcnn_checkpoint = 'model_data/detectron_model.pth'

cfg.merge_from_file('model_data/detectron_model.yaml')
cfg.freeze()

class VQAMaskRCNNBenchmark(nn.Module):
    def __init__(self):
        super(VQAMaskRCNNBenchmark, self).__init__()
        self.model = build_detection_model(cfg)

        model_state_dict = torch.load(maskrcnn_checkpoint)
        load_state_dict(self.model, model_state_dict.pop("model"))

        # make sure maskrcnn_benchmark is in eval mode
        self.model.eval()

    def _features_extraction(self, output,
                                 im_scales,
                                 feature_name='fc6',
                                 conf_thresh=0.5):
        batch_size = len(output[0]["proposals"])
        # list[num_of_boxes_per_image]
        n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
        # list[Tensor: (n_boxes_per_image, num_classes)]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        # list[Tensor: (n_boxes_per_image, 2048)]
        features = output[0][feature_name].split(n_boxes_per_image)
        # list[Tensor: (num_features_selected_per_image, 2048)]
        # list contain selected features per image
        features_list = []

        for i in range(batch_size):
            # reshape the bounding box to original size/coordinate
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            # Tensor: (n_boxes_per_image, num_classes)
            scores = score_list[i]
            # Tensor: (n_boxes_per_image, )
            # max_conf record the  heightest probs of the class
            # associate with each bounding box. If the heightest prob
            # of a box (say i) is smaller than threshold (conf_thresh), 
            # this box will not be select and max_conf[i] will be set
            # to 0 
            max_conf = torch.zeros((scores.shape[0])).to(device)

            for cls_ind in range(1, scores.shape[1]):
                # Tensor: (n_boxes_per_image, 1)
                # score for a specified class
                cls_scores = scores[:, cls_ind]
                # index of boxes that will be keep
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                            cls_scores[keep],
                                            max_conf[keep])
            
            # select the top 100 boxes which contain an onject with
            # probability greater than conf_thresh(usually 0.5)
            keep_boxes = torch.argsort(max_conf, descending=True)[:100]
            features_per_image = features[i][keep_boxes]
            features_list.append(features_per_image)
        
        return features_list

    def forward(self, images, image_scales):
        images = to_image_list(images, size_divisible=32)
        images = images.to(device)
        # the returned features of maskrcnn_benchmark is the result of
        # roi pooling without average pooling
        output = self.model(images)
        features = self._features_extraction(output, image_scales)

        return features