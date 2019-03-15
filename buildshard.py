"""Converts MSCOCO data to TFRecord file format with SequenceExample protos.

The MSCOCO images are expected to reside in JPEG files located in the following
directory structure:

    train/COCO_train2014_000000000151.jpg
    train/COCO_train2014_000000000260.jpg
    ...

The MSCOCO annotations JSON files are expected to reside in train_captions_file
and val_captions_file respectively.

This script converts the combined MSCOCO data into sharded data files consisting
of 196 TFRecord files:

    shard/train-00000-of-00192
    shard/train-00001-of-00192
    ...
    shard/train-00192-of-00192


Each TFRecord file contains ~1882 records. Each record within the TFRecord file
is a serialized SequenceExample proto consisting of precisely one image-caption
pair. Note that each image has multiple captions (usually 5) and therefore each
image is replicated multiple times in the TFRecord files.

The SequenceExample proto contains the following fields:

    context:
        image/image_id: integer MSCOCO image identifier
        image/data: string containing JPEG encoded image in RGB colorspace

    feature_lists:
        image/caption: list of strings containing the (tokenized) caption words
        image/caption_ids: list of integer ids corresponding to the caption words

The captions are tokenized using the NLTK (http://www.nltk.org/) word tokenizer.
The vocabulary of word identifiers is constructed from the sorted list (by
descending frequency) of word tokens in the training set. The vocabulary contains
5000 words according to the Config file

NOTE: This script will consume around 70GB of disk space because each image
in the MSCOCO dataset is replicated ~5 times (once per caption) in the output.
This is done for two reasons:
    1. In order to better shuffle the training data.
    2. It makes it easier to perform asynchronous preprocessing of each image in
         TensorFlow.

Running this script using 8 threads may take around 30 minutes on a Intel i7 4770.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
from copy import deepcopy
import json
import os.path
import random
import sys
import threading

import os
import pdb
import math
import numpy as np
import pandas as pd
from tqdm import tqdm


import nltk.tokenize
import tensorflow as tf

from utils.metrics.cocoset import COCO
from utils.vocabulary import Vocabulary

from config.config import Config

ImageMetadata = namedtuple("ImageMetadata",
            ["image_id", "filename", "caption", 'cls_lbls'])

try:
    xrange
except NameError:
    xrange = range

class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                                feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))

def _float_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def _float_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_float_feature(v) for v in values])

def _to_sequence_example(image, decoder, vocab, config):
    """Builds a SequenceExample proto for an image-caption pair.

    Args:
        image: An ImageMetadata object.
        decoder: An ImageDecoder object.
        vocab: A Vocabulary object.

    Returns:
        A SequenceExample proto.
    """

    with tf.gfile.FastGFile(image.filename, "r") as f:
        encoded_image = f.read()

    try:
        decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image.filename)
        return

    context = tf.train.Features(feature={
            "image/image_id": _int64_feature(image.image_id),
            "image/data": _bytes_feature(encoded_image),
    })

    current_caption_ids = vocab.process_sentence(image.caption)
    num_words = len(current_caption_ids)

    caption_ids = np.zeros(config.max_caption_length, dtype=np.int32)
    mask_ids  = np.zeros(config.max_caption_length, dtype=np.float32)
    mask_ids[:num_words] = 1.0
    caption_ids[:num_words] = np.array(current_caption_ids)
    
    if isinstance(image.cls_lbls, str):
        cls_lbls_str = image.cls_lbls[1:-1].replace(' ','').split(',')
        cls_lbls = [int(item) for item in cls_lbls_str if item != '']
    else:
        cls_lbls = image.cls_lbls#[1:-1].replace(' ','').split(',')
    
    one_hot_lbls = np.zeros([90])
    one_hot_lbls[cls_lbls] = 1
    one_hot_lbls = one_hot_lbls.astype(np.int64).tolist()
    # pdb.set_trace()
    feature_lists = tf.train.FeatureLists(feature_list={
            "image/caption_ids": _int64_feature_list(caption_ids),
            "image/mask_ids": _float_feature_list(mask_ids),
            "image/cls_lbls": _int64_feature_list(one_hot_lbls),
    })
    sequence_example = tf.train.SequenceExample(
            context=context, feature_lists=feature_lists)

    return sequence_example


def _process_image_files(thread_index, ranges, name, 
                                images, decoder, vocab, config):
    """Processes and saves a subset of images as TFRecord files in one thread.

    Args:
        thread_index: Integer thread identifier within [0, len(ranges)].
        ranges: A list of pairs of integers specifying the ranges of the dataset to
            process in parallel.
        name: Unique identifier specifying the dataset.
        images: List of ImageMetadata.
        decoder: An ImageDecoder object.
        vocab: A Vocabulary object.
        config: The configuration.
    """
    # Each thread produces N shards where N = num_shards / num_write_shard_threads.
    # For instance, if num_shards = 128, and num_write_shard_threads = 2, 
    # then the first thread would produce shards [0, 64).
    num_shards = config.num_shards
    num_write_shard_threads = len(ranges)
    assert not num_shards % num_write_shard_threads
    num_shards_per_batch = int(num_shards / num_write_shard_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                                num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in xrange(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(config.shard_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]

            sequence_example = _to_sequence_example(image, decoder, vocab, config)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                        (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
                    (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
                (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, image_ids, image_files, captions, cls_lbls,
                            vocab, config):

    """Processes a complete data set and saves it as a TFRecord.

    Args:
        name: Unique identifier specifying the dataset.
        images: List of ImageMetadata.
        vocab: A Vocabulary object.
        config: The configuration.
    """
    # Break up each image into a separate entity for each caption.
    num_shards = config.num_shards

    images = [ImageMetadata(image_ids[i], image_files[i], captions[i], cls_lbls[i])
                        for i in range(len(captions))]

    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(images)

    # Break the images into num_write_shard_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_write_shard_threads = min(num_shards, config.num_write_shard_threads)
    spacing = np.linspace(0, len(images), num_write_shard_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_write_shard_threads, ranges))
    for thread_index in xrange(len(ranges)):
        args = (thread_index, ranges, name, images, decoder, vocab, config)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
                (datetime.now(), len(images), name))

def main(unused_argv):

    config = Config()

    """ Prepare the data for training the model. """
    if not os.path.exists(config.prepare_annotation_dir):
        os.mkdir(config.prepare_annotation_dir)
    coco = COCO(config=config, 
                first_ann_file=config.train_caption_file, 
                second_ann_file=config.val_caption_file)
    
    print("Building the vocabulary...")
    vocabulary = Vocabulary(config.vocabulary_size)
    if not os.path.exists(config.vocabulary_file):
        coco.filter_by_cap_len(config.max_caption_length)
        vocabulary.build(coco.all_captions())
        vocabulary.save(config.vocabulary_file)
        vocabulary.save_counts(config.word_count_file)
    else:
        vocabulary.load(config.vocabulary_file)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))

    print("Processing the captions...")
    if not os.path.exists(config.train_csv_file):
        # pdb.set_trace()
        coco.filter_by_words(set(vocabulary.words))
        # assert len(coco.anns) == len(coco.cls_lbls)
        captions  = [coco.anns[ann_id]['caption']  for ann_id in coco.anns]
        image_ids = [coco.anns[ann_id]['image_id'] for ann_id in coco.anns]
        cls_lbls  = [coco.cls_lbls[ann_id]['cls_label'] for ann_id in coco.cls_lbls]
        assert len(captions) == len(image_ids)
        assert len(cls_lbls) == len(image_ids)

        image_files = [os.path.join(config.dataset_image_dir,
            'train' if coco.imgs[image_id]['file_name'].find('train2014') >= 0 else 'val',
            coco.imgs[image_id]['file_name']) for image_id in image_ids ] 

        annotations = pd.DataFrame({'image_id': image_ids,
                                    'image_file': image_files,
                                    'caption': captions,
                                    'cls_lbls': cls_lbls})
        annotations.to_csv(config.train_csv_file)
    else:
        annotations = pd.read_csv(config.train_csv_file)
        captions = annotations['caption'].values
        image_ids = annotations['image_id'].values
        image_files = annotations['image_file'].values
        cls_lbls = annotations['cls_lbls'].values
    

    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with config.num_write_shard_threads."""
        return num_shards < config.num_write_shard_threads or \
                             not num_shards % config.num_write_shard_threads

    assert _is_valid_num_shards(config.num_shards), (
        "Please make the config.num_writeshard_threads commensurate with config.train_shards")

    if not tf.gfile.IsDirectory(config.shard_dir):
        tf.gfile.MakeDirs(config.shard_dir)

        _process_dataset("train", image_ids, image_files, captions, cls_lbls,
                                vocabulary, config)
    else:
        print("The shards already exists")

if __name__ == "__main__":
    with tf.device('/cpu:0'):
        tf.app.run()
