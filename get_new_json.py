
import os
import pdb
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.vocabulary import Vocabulary
from config.config import Config

from cls_coco.utils.pycocotools.coco import COCO
import tqdm
from copy import deepcopy


def update_dict(kmeans, file_name):
    with open(file_name, 'r') as f:
        train_captions = json.load(f)
        train_captions_new = deepcopy(train_captions)

    cls_labels=[]
    skip_word = []

    annotations = tqdm.tqdm(train_captions['annotations'],
                    total=len(train_captions['annotations']))


    for idx, item in enumerate(annotations):
        caption = item[u'caption']
        labels = []
        for word in caption.split(' '):
            if word in v.words:
                word_index = v.word2idx[word]
                word_embed = word2vec[word_index]
                word_label = kmeans.predict(word_embed[np.newaxis,:])
                labels.append(word_label[0])
            else:
                skip_word.append(word)
        labels = list(set(labels))
        new_labels = []
        for label in labels:
            new_labels.append(int(label))

        tmp_dic = {u'image_id': item[u'image_id'], u'id': item[u'id'], u'cls_label':new_labels}
        cls_labels.append(deepcopy(tmp_dic))
    train_captions_new.update({'classifications':cls_labels})

    print('updated dict')
    file_name = file_name.split('/')[-1].split('.')[0]
    with open('./'+file_name+'_new.json',"w") as f:
        json.dump(train_captions_new, f)
        print(file_name+' saved')

def update_dict_with_instance(file_name):
    
    with open(file_name, 'r') as f:
        train_captions = json.load(f)
        train_captions_new = deepcopy(train_captions)

    instance_file = file_name.replace('captions', 'instances')
    coco = COCO(instance_file)

    import tqdm
    annotations = tqdm.tqdm(train_captions['annotations'],
                    total=len(train_captions['annotations']))
    cls_labels=[]
    # pdb.set_trace()
    
    for idx, item in enumerate(annotations):
        cls_id =[]
        annIds = coco.getAnnIds(imgIds=item[u'image_id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        for cls_item in anns:
            cls_id.append(cls_item['category_id']-1)
        cls_id = list(set(cls_id))
        tmp_dic = {u'image_id': item[u'image_id'], u'id': item[u'id'], u'cls_label':cls_id}
        cls_labels.append(deepcopy(tmp_dic))
    train_captions_new.update({'classifications':cls_labels})

  

    print('updated dict')
    file_name = file_name.split('/')[-1].split('.')[0]
    with open('./'+file_name+'_new.json',"w") as f:
        json.dump(train_captions_new, f)
        print(file_name+' saved')
        import shutil
        new_file = 'train.json' if 'train' in file_name else 'val.json'
        shutil.move('./'+file_name+'_new.json', 'datasets/rawjson/'+new_file)


if __name__ == '__main__':
    train_caption_file = './datasets/rawjson/captions_train2014.json'
    val_caption_file = './datasets/rawjson/captions_val2014.json'


    update_dict_with_instance(train_caption_file)

    update_dict_with_instance(val_caption_file)
    