import os
import os.path as osp
import sys
import random
import math
import numpy as np
import torch
import pickle
import PIL
from PIL import Image
import io

from torch.utils.data import Dataset

from .utils import convert_examples_to_features, read_examples
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from pytorch_pretrained_bert.tokenization import BertTokenizer
from .transforms import PIL_TRANSFORMS


# Meta Information
SUPPORTED_DATASETS = {
    'referit': {'splits': ('train', 'val', 'trainval', 'test')},
    'unc': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB', 'test'),
        'params': {'dataset': 'refcoco', 'split_by': 'unc'}
    },
    'unc+': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
    },
    'gref': {
        'splits': ('train', 'val'),
        'params': {'dataset': 'refcocog', 'split_by': 'google'}
    },
    'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
    },
    'flickr': {
        'splits': ('train', 'val', 'test')}
}

test_transforms = [
    dict(type='RandomResize', sizes=[640], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=640, center_place=True)
]

def processer(image, text, transforms=test_transforms):
    Transformer = []
    for t in transforms:
        _args = t.copy()
        Transformer.append(PIL_TRANSFORMS[_args.pop('type')](**_args))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    phrase = text.lower()
    target = {}
    target['bbox'] = torch.tensor([10, 10, 100, 100])
    target['phrase'] = phrase
    for transform in Transformer:
        image, target = transform(image, target)
    examples = read_examples(target['phrase'], 0)
    features = convert_examples_to_features(
        examples=examples, seq_length=40, tokenizer=tokenizer)
    word_id = features[0].input_ids
    word_mask = features[0].input_mask
    target['word_id'] = torch.tensor(word_id, dtype=torch.long)
    target['word_mask'] = torch.tensor(word_mask, dtype=torch.bool)
    if 'mask' in target:
        mask = target.pop('mask')
        return image, mask, target
    return image, target


