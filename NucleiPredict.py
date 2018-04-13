#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 05:29:03 2018

@author: briansp
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import numpy as np
import pandas as pd
import cv2

from config import Config
import utils
import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class NucleiConfig(Config):
    """Configuration for training on the nuclei dataset.
    Derives from the base Config class and overrides values specific
    to the nuclei dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nuclei"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 (nuclei)

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    #MEAN_PIXEL = np.array([122.5, 122.5, 122.5])
    #MINI_MASK_SHAPE = (84, 84)
    #MASK_SHAPE = [56, 56]

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128) # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 96

    #BACKBONE = "resnext101" # "resnet50", "resnet101", "resnext50", "resnext101"

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 940

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 32
    
    #USE_MINI_MASK = False
    MAX_GT_INSTANCES = 350 # may want to increase to 200

    # Normalizing while loading image
    #MEAN_PIXEL = np.array([0.,0.,0.])
	
    #LEARNING_RATE = 0.01 # try high learning rate, hoping for better result

config = NucleiConfig()
#config.display()

# get input image info
#IMG_WIDTH = 512
#IMG_HEIGHT = 512
IMG_CHANNELS = 3
DATA_PATH = '/root/input/'
#DATA_PATH = '/tmp/ramdisk/'
TRAIN_PATH_384 = DATA_PATH + 'stage1_train_384/'
TRAIN_PATH_512 = DATA_PATH + 'stage1_train_512/'
TRAIN_PATH_CLAHE = DATA_PATH + 'stage1_train_CLAHE/'
TEST_PATH = DATA_PATH + 'stage1_test/'


class NucleiDataset(utils.Dataset):
    """Provides the nuclei dataset.
    Images will be read and resized every time.
    Test images will not be resized.
    """
    def __init__(self, augment = True):
        super(NucleiDataset, self).__init__()
        self.masks = dict()
        self.augment = augment
        self.max_masks = 0
        self.file_path = None
        self.mean_pixel = [0, 0, 0]

    def load_nuclei(self, image_ids, shape, mean_pixel, file_path = TRAIN_PATH_512, alternate_path1 = TRAIN_PATH_CLAHE, alternate_path2 = None):
        """Generate and store the information about the data set.
        image_ids: list of image ids in the dataset.
        height, width: the size of the image data to be provided.
        file_path: directory where the files are stored.
        """
        # Add classes
        self.add_class("nuclei", 1, "nuclei")

        self.file_path = file_path
        self.image_height = shape[0]
        self.image_width = shape[1]
        self.mean_pixel = mean_pixel

        # Add images
        # Generate specifications of images (i.e. original image sizes
        # and locations). This is more compact than actual images.
        # Images are read in load_image() each time.
        for i, id_ in enumerate(image_ids):
            image_path = os.path.join(file_path, id_, 'images', id_ + '.png')
            alternate_image_path1 = os.path.join(alternate_path1, id_, 'images', id_ + '.png')
            if alternate_path2:
                alternate_image_path2 = os.path.join(alternate_path2, id_, 'images', id_ + '.png')
            else:
                alternate_image_path2 = None
            
            self.add_image("nuclei", image_id=i, path=image_path,
                           alternate_path1 = alternate_image_path1,
                           alternate_path2 = alternate_image_path2,
                           id_ = id_, height=self.image_height, width=self.image_width,)

    def load_image(self, image_id):
        """Load an image return it.
        TODO: apply CLAHE and store it for later use.
        """
        info = self.image_info[image_id]
        if self.augment:
            selection = np.random.random()
            if selection < .3:
                image_path = info['alternate_path1']
            elif selection < .6 and info['alternate_path2']:
                image_path = info['alternate_path2']
            else:
                image_path = info['path']
        else:
            image_path = info['path']

        img = cv2.imread(image_path)
        #mean = np.mean(img)
        #std = np.std(img)
        #img = (img-mean)/std + self.mean_pixel
        

        pad_height = None
        pad_width = None
        if img.shape[0] < self.image_height and img.shape[1] < self.image_width:
            new_img = np.zeros((self.image_height, self.image_width, IMG_CHANNELS))
            #print ('new image shape : ', new_img.shape)
            pad_height = (self.image_height - img.shape[0])//2
            pad_width = (self.image_width - img.shape[1])//2
            new_img[pad_height:pad_height + img.shape[0], pad_width:pad_width + img.shape[1], :] = img
            img = new_img

        if self.augment:
            transform_probability = 1 #np.random.random()
            perspective_probability = np.random.random()
            tf_matrix, perspective_matrix  = random_transform(img)
            img = apply_random_transform(img, transform_probability, tf_matrix,
                                         perspective_probability, perspective_matrix)
            img = random_channel_shift(img, 10, img_channel_axis)

        masks = []
        path = self.file_path + info['id_']
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
            if self.augment:
                mask_ = apply_random_transform(mask_, transform_probability, tf_matrix,
                                               perspective_probability, perspective_matrix)
            masks.append(mask_)

        # Record max number of masks
        self.max_masks = max(self.max_masks, len(masks))

        while True:
            if self.augment:
                origin_y = int(np.random.random()*((img.shape[0] - 1) // 64)) * 64
                origin_x = int(np.random.random()*((img.shape[1] - 1) // 64)) * 64

                origin_y = min(max(0, img.shape[0] - info['height']), origin_y)
                origin_x = min(max(0, img.shape[1] - info['width']), origin_x)
            else:
                origin_y = 0
                origin_x = 0

            info['origin_y'] = origin_y
            info['origin_x'] = origin_x

            if self.augment:
                end_y = origin_y+min(img.shape[0], info['height'])
                end_x = origin_x+min(img.shape[1], info['width'])
            else:
                end_y = img.shape[0]
                end_x = img.shape[1]

            info['end_y'] = end_y
            info['end_x'] = end_x

            partial_mask = []
            for mask_ in masks:
                if np.sum(mask_[origin_y:end_y, origin_x:end_x] > 1) > 4:
                    partial_mask.append(mask_[origin_y:end_y, origin_x:end_x] > 1)

            if len(partial_mask) > 0:
                self.masks[image_id] = partial_mask
                break
        
        return img[origin_y:end_y, origin_x:end_x, :]

    def load_test_image(self, image_id):
        """Generate a test image without resizing or augmentation.
        """
        info = self.image_info[image_id]
        img = cv2.imread(info['path'])
        #mean = np.mean(img)
        #std = np.std(img)
        #img = (img-mean)/std + self.mean_pixel
		
        return img
    
    def image_reference(self, image_id):
        """Return the information about the image."""
        info = self.image_info[image_id]
        if info["source"] == "nuclei":
            return info["nuclei"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Load annotated nuclei masks of the given image ID.
        """
        info = self.image_info[image_id]
        mask = self.masks.pop(image_id)
        count = len(mask)
        if count < 1:
            print ("Problem ", image_id, info['id_'])
        mask = np.stack(mask, axis = 2)
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index('nuclei') for i in range(count)])
        return mask, class_ids.astype(np.int32)


class InferenceConfig(NucleiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MAX_INSTANCES = 300
    DETECTION_MIN_CONFIDENCE = 0.90

# generate run length encoding of test data
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def mask_to_rles(masks):
    for i in range(masks.shape[2]):
        yield rle_encoding(masks[:,:,i])

def get_adjcent_model_name(model_name, increment):
    return model_name[:-7] + '%04d'%(int(model_name[-7:-3])+increment) + model_name[-3:]

def create_prediction(FLAGS):
    inference_config = InferenceConfig()

    # Validation dataset
    valid_df = pd.read_csv(DATA_PATH + 'stage1_val_data.csv')
    valid_ids = []
    for i, row in valid_df.iterrows():
        valid_ids.extend([row['ImageId']])
    print ("Number of validation images : ", len(valid_df))

    dataset_val = NucleiDataset(augment = False)
    dataset_val.load_nuclei(valid_ids, config.IMAGE_SHAPE, config.MEAN_PIXEL)
    dataset_val.prepare()

    # Test dataset
    test_ids = next(os.walk(FLAGS.test_dir))[1]

    dataset_test = NucleiDataset(augment = False)
    dataset_test.load_nuclei(test_ids, config.IMAGE_SHAPE, config.MEAN_PIXEL, FLAGS.test_dir)
    dataset_test.prepare()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
    # '/root/briansp/git/Mask_RCNN/logs/nuclei20180322T2343/mask_rcnn_nuclei_0014.h5' # LB .492

    # Load trained weights (fill in path to trained weights here)
    if FLAGS.use_swa:
        print("Averaging 3 weights around ", FLAGS.saved_model)
        weights = list()
        for i in range(-1, 2):
            model_name = get_adjcent_model_name(FLAGS.saved_model, i)
            print ('Processing ', model_name)
            model.load_weights(model_name, by_name=True)
            weights.append(model.keras_model.get_weights())
    
        new_weights = list()
        for weights_list_tuple in zip(*weights):
            new_weights.append(
                [np.array(weights_).mean(axis=0)\
                    for weights_ in zip(*weights_list_tuple)])
        model.keras_model.set_weights(new_weights)
        if FLAGS.sub_filename[-7:] != 'swa.csv':
            FLAGS.sub_filename = FLAGS.sub_filename[:-4] + '-swa.csv'
    else:
        model.load_weights(FLAGS.saved_model, by_name=True)
    '''
    APs = []
    for image_id in dataset_val.image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], tta = FLAGS.use_tta, verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.75)
        APs.append(AP)
    print("Validation mAP: ", np.mean(APs))
    '''

    new_test_ids = []
    rles = []
    max_masks = 0
    for image_id in dataset_test.image_ids:
        #print('Prediciting image id ', image_id)
        # Load image and ground truth data
        image_info = dataset_test.image_info[image_id]
        #print (image_info['path'])
        '''
        if image_info['id_'] in {
                '6710d57784372fd5557e1e0910ac296b4e384ca8f5505e6165aba5bb36fdf5b9',
                '9aa12685592e266c3ac3a355911ab684396a2a6d55d7453a9b8c416d64c3251b',
                '81537757f724d396ad6c803d8db775b2bd2814c711a839394e6895e5be62178d',
                '9079897a543f209c7272e854882037747a0d49f274a6e1ece18b04d23062757d',
                '6710d57784372fd5557e1e0910ac296b4e384ca8f5505e6165aba5bb36fdf5b9',
                '6710d57784372fd5557e1e0910ac296b4e384ca8f5505e6165aba5bb36fdf5b9',
                '6710d57784372fd5557e1e0910ac296b4e384ca8f5505e6165aba5bb36fdf5b9',
                '6710d57784372fd5557e1e0910ac296b4e384ca8f5505e6165aba5bb36fdf5b9',
                '6710d57784372fd5557e1e0910ac296b4e384ca8f5505e6165aba5bb36fdf5b9',
                '6710d57784372fd5557e1e0910ac296b4e384ca8f5505e6165aba5bb36fdf5b9',
                '6710d57784372fd5557e1e0910ac296b4e384ca8f5505e6165aba5bb36fdf5b9',
                '6710d57784372fd5557e1e0910ac296b4e384ca8f5505e6165aba5bb36fdf5b9'
            }:
            continue
        '''
        image = dataset_test.load_test_image(image_id)
        # Run object detection
        results = model.detect([image], tta = FLAGS.use_tta, verbose=0)
        r = results[0]
    
        # sort mask from small to large
        idx = list(enumerate(np.sum(r['masks'], axis = (0,1))))
        idx = sorted(idx, key = lambda x: x[1])
        idx = [x for x,y in idx]
        max_masks = max(max_masks, len(idx))
        r['masks'] = r['masks'][:,:,idx]
        if len(r['masks']) < 1:
            print ('No nuclei detected for ', image_info['id_'])
            rles.extend([[1,1]])
            new_test_ids.extend([image_info['id_']])
            continue

        #print (image_info['id_'], image.shape, r['masks'].shape, image.shape[0]*image.shape[1])
        # remove duplicate pixels
        mask = np.expand_dims(np.ones(image.shape[:2]), axis=2)
        for i in range(r['masks'].shape[2]):
            r['masks'][:,:,i:i+1] = (mask>.5) * r['masks'][:,:,i:i+1]
            mask[r['masks'][:,:,i:i+1] > .5] = 0

        # prepare data to make df
        rle = []
        rle = list(mask_to_rles(r['masks']))
        while [] in rle:
            print ("Removing empty mask for ", image_info['id_'])
            rle.remove([])
        rles.extend(rle)
        new_test_ids.extend([image_info['id_']] * len(rle))
        if FLAGS.test_output_dir:
            if not os.path.exists(FLAGS.test_output_dir):
                os.mkdir(FLAGS.test_output_dir)
            image_dir = os.path.join(FLAGS.test_output_dir, 'pseudo_' + image_info['id_'], 'images')
            masks_dir = os.path.join(FLAGS.test_output_dir, 'pseudo_' + image_info['id_'], 'masks')
            print(image_dir)
            os.makedirs(image_dir)
            cv2.imwrite(os.path.join(image_dir, 'pseudo_' + image_info['id_'] + '.png'), image)
            os.makedirs(masks_dir)
            for i in range(r['masks'].shape[2]):
                cv2.imwrite(os.path.join(masks_dir, 'pseudo_mask' + str(i) + '.png'), r['masks'][:,:,i].astype(np.uint8)*255)


    print ("Max_masks : ", max_masks)

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub['EncodedPixels'] = sub['EncodedPixels'].apply(lambda x: x if x != '1 1' else '')
    sub.to_csv(FLAGS.sub_filename, index=False)
    print ('Created ', FLAGS.sub_filename)
    print ("Done predicting nuclei")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--saved_model',
        type=str,
        default='logs/nuclei20180322T2343/mask_rcnn_nuclei_0014.h5',
        help='Location of saved model weights.')
    parser.add_argument(
        '--test_dir',
        type=str,
        default='/root/briansp/Downloads/nuclei/stage2_test_final',
        #default='/root/input/stage1_test',
        #default='/tmp/ramdisk/stage1_test',
        help='Location of test data files.')
    parser.add_argument(
        '--test_output_dir',
        type=str,
        default='',
        #default='/root/briansp/Downloads/nuclei/extra_data',
        help='Location to put predicted out so they can be used as pseudo labed data.')
    parser.add_argument(
        '--sub_filename',
        type=str,
        default='stage2-1-1.csv',
        help='Name of the submission file to create.')
    parser.add_argument(
        '--use_swa',
        type=bool,
        help='Whether to use SWA or not.')
    parser.add_argument(
        '--use_tta',
        type=bool,
        help='Whether to use test time augmentation or not.')

    FLAGS, unparsed = parser.parse_known_args()
    print ('FLAGS\n', FLAGS)

    create_prediction(FLAGS)

