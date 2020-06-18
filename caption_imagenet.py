import os
import json
import argparse
import torch
import time

from data_reader import DataReader
from dataset.imagenet_dataset import ImageNetDataset
from modules.vocabulary import Vocabulary
from modules.rcnn_encoder import VQAMaskRCNNBenchmark
from modules.captioner import PythiaCaptioner
from model.butd import PythiaBUTD
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CaptionImageNet:
    def __init__(self, root_dir, up_level, save_dir = 'captions', bad_words_dict = set()):
        encoder = VQAMaskRCNNBenchmark()
        captioner = PythiaCaptioner(use_constrained=True)
        self.model = PythiaBUTD(encoder=encoder, captioner=captioner)
        self.model.to(device)
        self.data_iterator = DataReader(root_dir)
        self.bad_words_dict = bad_words_dict
        self.up_level = up_level
        self.captions = {}
        self.save_dir = save_dir


    def caption(self):
        if self.data_iterator.has_next():
            img_dir = next(self.data_iterator)
            vocab = Vocabulary().get_vocabulary()
            dataset = ImageNetDataset(img_dir, vocab, up_levels=self.up_level, bad_words_dict=self.bad_words_dict)
            wordnet_id = os.path.split(img_dir)[-1]
            print('\twordnet id: {}'.format(wordnet_id))
            if (os.path.exists(os.path.join(self.save_dir, wordnet_id + '.json'))):
                print('\t\tSynset has been already processed')
                return
            # caption for single synset
            img_to_caption = {}
            self.captions[wordnet_id] = []
            for item_ind in tqdm(range(len(dataset))):
                time.sleep(0.01)
                # get unique image id
                image_id = dataset.get_img_id(item_ind)
                # get the data fields
                try:
                    data = dataset[item_ind]
                except:
                    continue
                image = [data['image']]
                image_scale = [data['image_scale']]
                transitions = data['transitions']
                # calculate caption
                caption = self.model(image, image_scale, transitions=transitions)
                img_to_caption[image_id] = caption
                self.captions[wordnet_id].append({"id": image_id, "capation": caption})
                json_file = wordnet_id + '.json'
                json_path = os.path.join(self.save_dir, json_file)
                # write to json file
                self.write_to_json(json_path, img_to_caption)

    def write_to_json(self, json_path, source):
        with open(json_path, 'w') as j:
            json.dump(source, j)

    def run_all(self):
        count = 1
        while (self.data_iterator.has_next()):
            print('Process {}th synset:'.format(count))
            self.caption()
            count += 1

parser = argparse.ArgumentParser(description='Process data directory')
parser.add_argument('--root_dir', type=str)
parser.add_argument('--up_level', type=int)
parser.add_argument('--save_dir', type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    root_dir = args.root_dir
    up_level = args.up_level
    save_dir = args.save_dir
    bad_words_dict = ('being')
    proc = CaptionImageNet(root_dir, up_level, save_dir=save_dir, bad_words_dict=bad_words_dict)
    proc.run_all()