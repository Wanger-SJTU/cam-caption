
# coding: utf-8

# In[19]:

import os
import json
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models.nnets import NN
from utils.vocabulary import Vocabulary
from config.config import Config
from models.models import ShowAttendTell


from copy import deepcopy


def update_dict(file_name):
    with open(file_name, 'r') as f:
        train_captions = json.load(f)
        train_captions_new = deepcopy(train_captions)

    print('*'*30)
    print('updating dict '+ file_name)

    cls_labels=[]
    skip_word = []
    import tqdm
    annotations = tqdm.tqdm(train_captions['annotations'],
                    total=len(train_captions['annotations']))

    for idx, item in enumerate(annotations):
        caption = item[u'caption']
        labels = []
    #     print(caption)
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

    print('update dict')
    file_name = file_name.split('.')[0]
    with open('./'+file_name+'_new.json',"w") as f:
        json.dump(train_captions_new, f)
        print('saved')

if __name__ == '__main__':
    config = Config()
    config.train_cnn = False
    config.phase = 'train'
    nn = NN(config)


    # In[3]:

    model = ShowAttendTell(config)
    # model.build()


    # In[4]:

    v = Vocabulary(7300)
    v.load('./datasets/vocabulary.csv')
    print(v.words.shape)
    print((v.word2idx[v.words[1]]))


    # In[5]:

    word = tf.placeholder(tf.int32, shape=[1])
    with tf.variable_scope("word_embedding",reuse=tf.AUTO_REUSE):
        embedding_matrix = tf.get_variable(
                    name = 'weights',
                    shape = [7300, 512],
                    initializer = nn.fc_kernel_initializer,
                    regularizer = nn.fc_kernel_regularizer,
                    trainable   = True)
        word_embed = tf.nn.embedding_lookup(embedding_matrix, word)


    # In[6]:

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)


    # In[7]:

    include = ['word_embedding/weight']
    variables_to_restore = slim.get_variables_to_restore(include=include)
    # variables_to_restore = slim.get_variables(scope="word_embedding")
    word_embed_list = []

    with tf.Session() as sess:
        checkpoint_path = tf.train.latest_checkpoint('./results/checkpoint/')
        print(checkpoint_path)
        saver = tf.train.Saver(variables_to_restore)
        tf.contrib.framework.get_variables_to_restore()
        saver.restore(sess, checkpoint_path)
        word2vec = embedding_matrix.eval()


    # In[10]:

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=256,
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




    update_dict(train_caption_file)

    update_dict(val_caption_file)
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



