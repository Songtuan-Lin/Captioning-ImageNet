import torch

class TableTensor:
    def __init__(self, vocab, tables):
        '''
        Convert a finite automata transition table to Pytorch tensors
        Args:
            vocab (Pythia.Vocabulary): a pre-defined Pythia vocabulary
            tables (Dict{int: Dict{int: list[str]}}): transition table
            of finite automata
        '''
        self.tables = tables
        self.vocab = vocab

    def __to_tensor(self, v):
        '''
        Convert a small transition table for state v to Pytorch tensor
        Args:
            v (int): vertex
        Returns:
            Tensor: a tensor T with size (num_of_states, vocab_size),
            in which, T[j, k] = 0 indicates state j can transist to 
            state v by consuming the kth token in vocabulary and
            T[j, k] = 1 means otherwise.
        '''
        table = self.tables[v]
        table_tensor = torch.ones(len(self.tables), self.vocab.get_size())
        for v, token in table.items():
            if token == '.':
                table_tensor[v] = 0
            else:
                if token in self.vocab.get_stoi():
                    token_id = self.vocab.get_stoi()[token]
                    table_tensor[v][token_id] = 0
        return table_tensor.bool()

    def to_tensors(self):
        '''
        Convert the entire transition table of finite automata
        to a list of Pytorch tensor
        Returns:
            list[Tensor]: the ith tensor in the list is the 
            transition table tensor of state i (obtained by 
            calling __to_tensor())
        '''
        tensors = {}
        for v in range(len(self.tables)):
            tensors[v] = self.__to_tensor(v)
        return tensors