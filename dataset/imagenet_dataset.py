import os
import cv2
import torch
import json
import inflect
import numpy as np

from PIL import Image

from torch.utils.data import Dataset
from nltk.corpus import wordnet as wn


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class ImageNetDataset(Dataset):
    def __init__(self, root, vocab, up_levels=2, bad_words_dict = set()):
        if 'images' in os.listdir(root):
            root = os.path.join(root, 'images')
        image_paths = os.listdir(root)
        self.image_paths = [os.path.join(root, path) for path in image_paths]
        self.bad_words_dict = bad_words_dict
        # get the corresponding synset tag and use it
        # build transition table
        wordnet_id = os.path.split(root)[-1]
        # synset_tag = self._get_synset_tag(wordnet_id)
        synset_tag = self._get_hypernyms(wordnet_id, target_level=up_levels)
        self.transitions = self._build_transitions(synset_tag, vocab)
        

    def _get_hypernyms(self, wordnet_id, target_level):
        '''
        get all hypernyms of a specific wordnet id

        Args:
            wordnet_id (str): a wordnet id

        Returns:
            List[str]: a list of phrases(lemma names) which are the hypernyms 
            of input wordner id
        '''

        def get(syns, current_level, target_level):
            '''
            a helper function which get all the hypernyms of input synsets recursively

            Args:
                syns: the list input synsets
            Returns:
                hypers: the list of all hypernyms
            '''
            if syns == [] or current_level == target_level:
                # if the current synset is empty, finish search
                return []
            else:
                hypers = []
                for syn in syns:
                    # get the hypernyms of each synset in the list
                    hypers += syn.hypernyms()
                # get the hypernyms of current hypernyms
                hypers += get(hypers, current_level + 1, target_level) 

                return hypers

        syn = wn.synset_from_pos_and_offset(wordnet_id[0], int(wordnet_id[1:]))
        hypers = get([syn], current_level=0, target_level=target_level)
        phrases = syn.lemma_names()
        for hyper in hypers:
            phrases = phrases + hyper.lemma_names()

        for idx, phrase in enumerate(phrases):
            phrases[idx] = phrase.replace('_', ' ')

        for idx, phrase in enumerate(phrases):
            phrases[idx] = phrase.lower()

        phrases = self._clean(phrases, self.bad_words_dict)
        phrases = self._pluralise(phrases)

        return phrases

    def _clean(self, phrases, bad_words_dict):
        cleaned_phrases = []
        for phrase in phrases:
            if phrase not in bad_words_dict:
                cleaned_phrases.append(phrase)
        return cleaned_phrases

    def _pluralise(self, phrases):
        p = inflect.engine()
        plural = []
        for phrase in phrases:
            if len(phrase.split(' ')) == 1:
                plural.append(p.plural(phrase))
        return phrases + plural

    def _get_synset_tag(self, wordnet_id):
        '''
        get synset tag from wordnet id

        Args:
            wordnet_id (str): wordnet id of a synset tag

        Returen:
            List[str]: synset tag which is a list of phrases
        '''
        synset = wn.synset_from_pos_and_offset(wordnet_id[0], int(wordnet_id[1:]))
        synset_tag = synset.lemma_names()
        for idx, phrase in enumerate(synset_tag):
            synset_tag[idx] = phrase.replace('_', ' ')

        return synset_tag

    def _count_beam_num(self, synset_tag, vocab):
        beam_num = 2  # count init and accept state in
        for phrase in synset_tag:
            words = phrase.split()
            if len(words) > 1:
                for word in words:
                    if word not in vocab.get_stoi():
                        break
                    if word != words[-1]:
                        beam_num += 1
        return beam_num

    def _build_transitions(self, synset_tag, vocab):
        '''
        build state machine transition table using synset tag, for each
        state, we build a Tensor with shape (beam_num, vocab_size) to record
        transition, each row represent a state/beam, a column which marked 
        0 means it can trigger the state transition and 1 means forbidden,
        i.e. if slot (1, 15) is 0, then, word-id 15 can trigger state 
        transition from state 1 to current state. 

        Args:
            synset_tag (List[str]): synset tag contains a list of phrases
            vocab (PythiaVocab): Pythia vocab object

        Return:
            List[torch.Tensor]: 
        '''
        # get the vocab size and total number of beams/states
        vocab_size = vocab.get_size()
        beam_num = self._count_beam_num(synset_tag, vocab)
        transitions = {}
        # initialize transition for init state
        init_transition = torch.zeros(beam_num, vocab_size)
        init_transition[-1] = 1
        # initialize transition for accept state
        accept_transition = torch.ones(beam_num, vocab_size)
        accept_transition[-1] = 0
        # assign idx 0 to init state and beam_num - 1 to accept state
        transitions[0] = init_transition
        transitions[beam_num - 1] = accept_transition
        # number of middle state
        middle_state_num = 0
        # iterate through each phrase in synset tag
        for phrase in synset_tag:
            # split phrase into words
            words = phrase.split()
            # the initial state for each start of 
            # iteration is always init 
            prev_state = 0
            # iterate through each word
            for idx, word in enumerate(words):
                if word not in vocab.get_stoi():
                    break
                # get the word-id from vocab
                word_id = vocab.get_stoi()[word]
                # the case we reach the final word of phrase
                if idx == len(words) - 1:
                    # current state should be accept state
                    current_state = beam_num - 1
                    # current word can NOT trigger state transition
                    # prev_state-init, and hence, the slot should be marked
                    # as 1
                    transitions[0][prev_state, word_id] = 1
                    # current can trigger state transition prev_state-accept
                    transitions[current_state][prev_state, word_id] = 0
                # the case we have not reached the end word of phrase
                else:
                    # in this case, we need middle state
                    middle_state_num += 1
                    # we index current state as the numeber of middle states
                    current_state = middle_state_num
                    # initialize transition table for this middle state
                    transitions[current_state] = torch.ones(beam_num, vocab_size)
                    # current word can trigger state transition
                    # prev_state-current_state
                    transitions[current_state][prev_state, word_id] = 0
                    # current word can NOT trigger state transition
                    # prev_state-init
                    transitions[0][prev_state, word_id] = 1
                    # update prev_state with current state
                    prev_state = current_state

        transitions = [transitions[i].bool() for i in range(beam_num)]
        transitions = [transition.to(device) for transition in transitions]

        return transitions

    def _image_transform(self, image_path):
        '''
        Read an image and apply necessary transform

        Args:
            image_path (str): path to the image

        Returns:
            Tensor: image data
            int: scale used to resize image
        '''
        img = Image.open(image_path)
        im = np.array(img).astype(np.float32)
        if len(im) == 2:
            im = im[:,:,np.newaxis]
            im = np.concatenate((im,im,im), axis=2)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(800) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > 1333:
            im_scale = float(1333) / float(im_size_max)
        im = cv2.resize(
            im,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)
        return img, im_scale 

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image_raw = Image.open(image_path)
        image, image_scale = self._image_transform(image_path)
        data = {'image_raw': image_raw, 'image': image, 
                'image_scale': image_scale, 'transitions': self.transitions}
        
        return data

    def __len__(self):
        return len(self.image_paths)

    def get_img_id(self, item_ind):
        return os.path.split(self.image_paths[item_ind])[-1]