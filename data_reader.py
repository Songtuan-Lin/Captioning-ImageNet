import os

from collections import Iterator

class DataReader(Iterator):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dirs = os.listdir(root_dir)
        self.ptr = 0

    def has_next(self):
        return self.ptr < len(self.img_dirs)

    def __next__(self):
        if self.has_next():
            img_dir = self.img_dirs[self.ptr]
            dir = os.path.join(self.root_dir, img_dir)
            self.ptr += 1
            return dir
        raise StopIteration
