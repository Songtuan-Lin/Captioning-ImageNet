import torch

from pythia.common.registry import registry
from pythia.utils.text_utils import TextDecoder


@registry.register_decoder("constrained_beam_search")
class ConstrainedBeamSearch(TextDecoder):
    def __init__(self, vocab, config):
        super(ConstrainedBeamSearch, self).__init__(vocab)
        # beam size
        self._decode_size = config["inference"]["params"]["beam_length"]

    def init_batch(self, sample_list):
        # constrained beam search must have transition table
        assert hasattr(sample_list, 'transitions')
        # get the transitions and number of states
        self.transitions = sample_list.transitions
        self._beam_num = len(self.transitions)
        # use a list to hold the actural number of 
        # candidates in each beam, initial value for
        # each beam is the beam size
        self._beam_sizes = [self._decode_size for _ in range(self._beam_num)]
        # record joint probability of each sequence
        setattr(self, 'top_scores', 
                sample_list.answers
                .new_zeros(
                    (self._decode_size * self._beam_num, 1), 
                    dtype=torch.float)
                )
        # record selected sequence in each beam
        self.seqs = sample_list.answers.new_full(
            (self._decode_size * self._beam_num, 1), self._vocab.SOS_INDEX, 
            dtype=torch.long)
        self.seqs = list(self.seqs.split(self._beam_sizes))
        # copy the image feature to shape 
        # (beam_num * beam_size, num_boxes, feature_size)
        sample_list.image_feature_0 = (
            sample_list.image_feature_0.unsqueeze(1)
            .expand(-1, self._decode_size * self._beam_num, -1, -1)
            .squeeze(0)
        )

        return sample_list

    def _update_states(self, data, prev_word_idx, incomplete_idx):
        # fetch the corresponding hidden state base on the indices
        # of selected words
        h1 = data["state"]["td_hidden"][0][prev_word_idx[incomplete_idx]]
        c1 = data["state"]["td_hidden"][1][prev_word_idx[incomplete_idx]]
        h2 = data["state"]["lm_hidden"][0][prev_word_idx[incomplete_idx]]
        c2 = data["state"]["lm_hidden"][1][prev_word_idx[incomplete_idx]]

        states = {'h1': h1, 'c1': c1, 'h2': h2, 'c2': c2}
        return states

    def _update_seqs(self, seq, seqs, prev_word_idx, next_word_idx):
        seqs = torch.cat(seqs)
        return torch.cat([seqs[prev_word_idx], next_word_idx.unsqueeze(1)], dim=1)


    def decode(self, t, data, scores):
        '''
        Perform single step beams update
        Args:
            t (int): time step
            data (Dict{str: Tensor}): contains information required in beam update,
            e.g. state input and tokens selected in last time step
            scores (Tensor): local probability distribution produced at last time step,
            i.e. p(w(t - 1) | w1 w2 ... w(t - 2), I)
        Returns:
            Dict{str: Tensor}: updated data information
            int: total number of sequences which have not reach EOS token
        '''
        # shape: (beam_num * beam_size, vocab_size)
        scores = torch.nn.functional.log_softmax(scores, dim=1)
        # update the joint proability
        scores = self.top_scores.expand_as(scores) + scores
        # if it is the first timestep, we only take the first prediction
        if t == 0:
            scores = scores[0]
        # list to store joint probability of sequence in each beam
        top_scores = []
        # list to store hidden state for each beam
        h1, c1, h2, c2 = [], [], [], []
        # list to store predicted candidates for each beam
        texts = []
        seqs = [seq.clone() for seq in self.seqs]
        # create shallow list to store beam sizes
        beam_sizes = self._beam_sizes[:]
        # iterate through each beam
        for beam_idx in range(self._beam_num):
            # shape: (beam_num, vocab_size)
            transition = self.transitions[beam_idx].clone()
            # at first timestep, only init state is valid
            if t == 0:
                transition = transition[0]
            else:
                # split transition table of each beam
                splits = list(transition.split(1))
                # broadcast transition table
                for i, split in enumerate(splits):
                    splits[i] = split.expand(self._beam_sizes[i], -1)
                transition = torch.cat(splits)
            # mask the words which cannot trigger state transition with -inf
            beam_scores = scores.masked_fill(transition, float('-inf'))
            top_beam_scores, top_beam_words = beam_scores.view(-1).topk(
                self._beam_sizes[beam_idx], 0, True, True
            )
            # previous sequence
            prev_word_idx = top_beam_words // self._vocab_size
            # predict word idx
            next_word_idx = top_beam_words % self._vocab_size
            # update sequence
            self.seqs[beam_idx] = self._update_seqs(
                self.seqs[beam_idx], seqs, prev_word_idx, next_word_idx
            )
            # find the complete and incomplete sequence within current beam
            complete_idx, incomplete_idx = self.find_complete_inds(next_word_idx)
            # if accept state has sequence reach <eos> token, record them
            if len(complete_idx) > 0 and beam_idx == self._beam_num - 1:
                self._complete_seqs.extend(self.seqs[beam_idx][complete_idx].tolist())
                self._complete_seqs_scores.extend(top_beam_scores[complete_idx])
            # remove completed sequences
            self.seqs[beam_idx] = self.seqs[beam_idx][incomplete_idx]
            # update beam size
            beam_sizes[beam_idx] -= len(complete_idx)
            # remove the joint probability of complete sequence in current beam
            top_beam_scores = top_beam_scores[incomplete_idx].unsqueeze(1)
            # append current joint probability to list
            top_scores.append(top_beam_scores)
            # record hidden states for selected candidates
            beam_states = self._update_states(data, prev_word_idx, 
                                              incomplete_idx)
            h1.append(beam_states['h1'])
            c1.append(beam_states['c1'])
            h2.append(beam_states['h2'])
            c2.append(beam_states['c2'])
            # record predicted candidates
            texts.append(next_word_idx[incomplete_idx].unsqueeze(1))
        self._beam_sizes = beam_sizes
        # update texts field, which will be the next input to model
        data['texts'] = torch.cat(texts)
        # update hidden states
        h1 = torch.cat(h1)
        c1 = torch.cat(c1)
        h2 = torch.cat(h2)
        c2 = torch.cat(c2)
        data["state"] = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}
        # update joint probability for all sequences
        self.top_scores = torch.cat(top_scores)

        if self._beam_sizes[-1] == 0:
            return True, data, 0
        else:
            next_batch_total_size = sum(self._beam_sizes)
            return False, data, next_batch_total_size

    def get_result(self):
        if len(self._complete_seqs_scores) == 0:
            captions = torch.FloatTensor([0] * 5).unsqueeze(0)
        else:
            i = self._complete_seqs_scores.index(max(self._complete_seqs_scores))
            captions = torch.FloatTensor(self._complete_seqs[i]).unsqueeze(0)
        return captions
