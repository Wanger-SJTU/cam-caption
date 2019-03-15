import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import pandas as pd
import skimage.transform

from PIL import Image


import tensorflow as tf
import numpy as np

from caption.caption import CaptionGenerator
from config.config import Config
from models.models import ShowAttendTell
from utils.dataprovider import DataProvider
from utils.vocabulary import Vocabulary

FLAGS = tf.app.flags.FLAGS


tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

def visualize_att(image_file, seq, alpha_sent, vocab, config, smooth=False):
    """
      Visualizes caption with weights at every word.
      Adapted from paper authors' repo:
      https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """

    image_name = image_file.split(os.sep)[-1]
    image_name = os.path.splitext(image_name)[0]

    image = Image.open(image_file)

    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)
    image = skimage.img_as_float(image)
    word_idxs = [0]+seq
    words = [vocab.words[ind] for ind in word_idxs]

    for t in range(len(word_idxs)):
        plt.subplot(np.ceil(len(word_idxs) / 5.), 5, t+1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=8)
        plt.imshow(image)
        current_alpha = alpha_sent[t].reshape(14,14)
        '''
        if t > 0:

            t_min = np.min(current_alpha)
            t_max = np.max(current_alpha)
            current_alpha = (current_alpha - t_min) / (t_max - t_min)
            current_alpha = 1 - current_alpha
        '''
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.6)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')    
    plt.savefig(os.path.join(config.infer_result_dir,
                                         'v_' + image_name+'.jpg'))
    plt.close()


def main(argv):
    print("Testing the model ...")
    config = Config()
    config.phase = 'infer'

    config.beam_size = FLAGS.beam_size
    if not os.path.exists(config.infer_result_dir):
        os.mkdir(config.infer_result_dir)
    print("Building the vocabulary...")
    vocabulary = Vocabulary(config.vocabulary_size)
    vocabulary.load(config.vocabulary_file)
    print("Vocabulary built.")
    print("Number of words = %d" %(vocabulary.size))



    infer_data = DataProvider(config)
    model = ShowAttendTell(config)
    model.build()    
    

    with tf.Session() as sess:      
        model.setup_graph_from_checkpoint(sess, config.checkpoint_dir)
        tf.get_default_graph().finalize()

        captiongen = CaptionGenerator(model,
                                   vocabulary,
                                   config.beam_size,
                                   config.max_caption_length,
                                   config.batch_size)
        captions = []
        scores = []

        # Generate the captions for the images
        for k in tqdm(list(range(infer_data.num_batches)), desc='path'):
            batch,images = infer_data.next_batch_and_images()
            caption_data = captiongen.beam_search(sess, images,vocabulary)

            fake_cnt = 0 if k<infer_data.num_batches-1 \
                         else infer_data.fake_count
            for l in range(infer_data.batch_size-fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                alpha_sent = caption_data[l][0].alpha

                caption = vocabulary.get_sentence(word_idxs)
                captions.append(caption)
                scores.append(score)

                # Save the result in an image file
                image_file = batch[l]
                visualize_att(image_file,word_idxs,alpha_sent,vocabulary,config)
                image_name = image_file.split(os.sep)[-1]
                image_name = os.path.splitext(image_name)[0]
                img = plt.imread(image_file)
                plt.imshow(img)
                plt.axis('off')
                plt.title(caption)
                plt.savefig(os.path.join(config.infer_result_dir,
                                         image_name+'_result.jpg'))

        # Save the captions to a file
        results = pd.DataFrame({'image_files':infer_data.image_files,
                                'caption':captions,
                                'prob':scores})
        results.to_csv(config.infer_result_file)
    print("Testing complete.")
            

if __name__ == '__main__':
    tf.app.run()
