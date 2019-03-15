
# coding: utf-8

# In[19]:

import os
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils.vocabulary import Vocabulary
from config.config import Config

from cls_coco.utils.pycocotools.coco import COCO

from copy import deepcopy


def update_dict(kmeans, file_name):
    with open(file_name, 'r') as f:
        train_captions = json.load(f)
        train_captions_new = deepcopy(train_captions)

    cls_labels=[]
    skip_word = []
    import tqdm
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

    for idx, item in enumerate(annotations):
        cls_id = []
        annIds = coco.getAnnIds(imgIds=item[u'image_id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        for item in anns:
            cls_id.append(item['category_id']-1)

        tmp_dic = {u'image_id': item[u'image_id'], u'id': item[u'id'], u'cls_label':cls_id}
        cls_labels.append(deepcopy(tmp_dic))
    train_captions_new.update({'classifications':cls_labels})

    print('updated dict')
    file_name = file_name.split('/')[-1].split('.')[0]
    with open('./'+file_name+'_new.json',"w") as f:
        json.dump(train_captions_new, f)
        print(file_name+' saved')


if __name__ == '__main__':
    config_tf = tf.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    config = Config()
    config.train_cnn = False
    config.phase = 'train'
   
    v = Vocabulary(5000)
    v.load('./datasets/vocabulary.csv')
    print(v.words.shape)
    print((v.word2idx[v.words[1]]))
    data_path = './289999.npy'
    print("Loading the CNN from %s..." %data_path)
    data_dict = np.load(data_path).item()
    count = 0
    for op_name in tqdm(data_dict):
        if 'word_embedding/weights' in op_name:
            word2vec = data_dict[op_name]
      
   

    # In[10]:

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=512,
                    init='k-means++', 
                    n_init=10, 
                    max_iter=3000, 
                    tol=0.0001, 
                    precompute_distances='auto', 
                    verbose=0, 
                    random_state=None, 
                    copy_x=True, 
                    n_jobs=1, 
                    algorithm='auto')


    # In[11]:
    print('-'*20)
    print('clustering')
    print('-'*20)
    kmeans.fit(word2vec[1:])
    print('-'*20)
    print('clustering done')
    print('-'*20)

    import numpy as np 


    train_caption_file = './datasets/rawjson/captions_train2014.json'
    val_caption_file = './datasets/rawjson/captions_val2014.json'


    update_dict(kmeans, train_caption_file)

    update_dict(kmeans, val_caption_file)
    # word_cls ={}
    # for word in v.words:
    #     idx = v.word2idx[word]
    #     embeded = word2vec[idx][np.newaxis,:]
    #     label = kmeans.predict(embeded)[0]
    #     if label in word_cls.keys():
    #         word_cls[label].append(word)
    #     else:
    #         word_cls.update({label:[word]})



    # for key in word_cls.keys():
    #     print(str(key))
    #     print(word_cls[key])


    # # In[ ]:



