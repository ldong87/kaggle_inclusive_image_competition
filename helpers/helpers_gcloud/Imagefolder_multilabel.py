import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        img = accimage_loader(path)
        return img
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        # print(path)
        return accimage_loader(path)
    else:
        # print(path)
        return pil_loader(path)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(dir, label_file, desc_file, extensions):
    import pandas as pd
    label = pd.read_csv(label_file) # train or test
    
    label_desc = pd.read_csv(desc_file)
    classes = list(label_desc['label_code'])
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = dict((v,k) for k,v in class_to_idx.items())
    class_to_desc = label_desc.set_index('label_code') 
    class_to_desc = class_to_desc.to_dict()['description']
    weights_ = list(label_desc['weight_class_'])

    images = []
    dir = os.path.expanduser(dir)
    images = list(zip(dir+label['ImageID']+'.jpg', label['label_code'].map(lambda x: [class_to_idx[i] for i in x.split(',')]), label['weight_class']  ))
    weights = list(label['weight_class'].map(lambda x: int(x)))
    num_samples = label['weight_class'].unique().sum()
    return images, weights, classes, class_to_idx, idx_to_class, class_to_desc, int(num_samples), weights_


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, label_file, desc_file, extensions=IMG_EXTENSIONS, loader=default_loader, transform=None, target_transform=None):
        samples, weights, classes, class_to_idx, idx_to_class, class_to_desc, num_samples, weights_ = make_dataset(root, label_file, desc_file, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.class_to_desc = class_to_desc
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.weights = weights
        self.weights_ = weights_
        self.num_samples = num_samples
        self.transform = transform
        self.target_transform = target_transform
        self.label_file = label_file

        from sklearn.preprocessing import MultiLabelBinarizer as MLB
        self.mlb_fit = MLB().fit([range(len(self.classes))])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target, weight = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        #if self.target_transform is not None:
        #    target = self.target_transform(target)
        
        import numpy as np
        target = np.squeeze(self.mlb_fit.transform([target]))
        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


'''
class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
'''
