#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 05:29:03 2018

@author: briansp
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
import math

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
print ("ROOT_DIR : ", ROOT_DIR)
print ("MODEL_DIR : ", MODEL_DIR)
print ("COCO_MODEL_PATH : ", COCO_MODEL_PATH)

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

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128) # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    #BACKBONE = "resnet50" # "resnet50", "resnet101", "resnext50", "resnext101"

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 880
    #STEPS_PER_EPOCH = 1070 #2139

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 32
    
    MAX_GT_INSTANCES = 350 # may want to increase to 200

    # Normalizing while loading image
    #MEAN_PIXEL = np.array([0.,0.,0.])
	
    LEARNING_RATE = 0.002 # try high learning rate, hoping for better result
    #LEARNING_MOMENTUM = .86

config = NucleiConfig()
config.display()

# get input image info
#IMG_WIDTH = 512
#IMG_HEIGHT = 512
IMG_CHANNELS = 3
#DATA_PATH = '/root/briansp/Download/nuclei/'
DATA_PATH = '/root/input/'
#DATA_PATH = '/tmp/ramdisk/'
TRAIN_PATH_384 = DATA_PATH + 'stage1_train_384/'
TRAIN_PATH_512 = DATA_PATH + 'stage1_train_512/'
TRAIN_PATH_CLAHE = DATA_PATH + 'stage1_train_CLAHE/'
TEST_PATH = DATA_PATH + 'stage1_test/'

# Get train and test IDs
train_df = pd.read_csv(DATA_PATH + 'stage1_train_data.csv')
train_ids = []
for i, row in train_df.iterrows():
    if row['ImageId'] != '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80': # bad labeling
        train_ids.extend([row['ImageId']] * int(row['mult_factor']))
extra_ids = next(os.walk(TRAIN_PATH_512))[1]
extra_ids = list(filter(lambda name: name[:4] == 'TCGA' or name[:7] == 'pseudo_', extra_ids))*4
extra_ids = None
#train_ids.extend(extra_ids)
print ("Number of training images (multiplied by boost factor) : ", len(train_ids))
#print ("Number of external/pseudo training images (multiplied by boost factor) : ", len(extra_ids))

valid_df = pd.read_csv(DATA_PATH + 'stage1_val_data.csv')
valid_ids = []
for i, row in valid_df.iterrows():
    valid_ids.extend([row['ImageId']])
print ("Number of validation images : ", len(valid_df))

#mask_df = pd.read_csv(DATA_PATH + 'stage1_train_labels.csv')

test_ids = next(os.walk(TEST_PATH))[1]

img_row_axis = 0
img_col_axis = 1
img_channel_axis = 2

def random_channel_shift(x, intensity, channel_axis=0):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [
        np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x,
                max_x) for x_channel in x
    ]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    return cv2.warpAffine(x, transform_matrix[:2, :], (x.shape[1], x.shape[0]))

def random_transform(img, rotation_range = 5, height_shift_range = .05, width_shift_range = .05,
                     shear_range = 5, zoom_range = (.90, 1.10),
                     perspective_range = 0, seed=None):
    """Randomly augment a single image tensor.
    # Arguments
        x: 3D tensor, single image.
        seed: random seed.
    # Returns
        A randomly transformed version of the input (same shape).
    """
    if seed is not None:
        np.random.seed(seed)

    # use composition of homographies
    # to generate final transform that needs to be applied
    if rotation_range:
        theta = np.deg2rad(np.random.uniform(-rotation_range, rotation_range))
    else:
        theta = 0

    if height_shift_range:
        tx = np.random.uniform(-height_shift_range, height_shift_range)
        if height_shift_range < 1:
            tx *= img.shape[img_row_axis]
    else:
        tx = 0

    if width_shift_range:
        ty = np.random.uniform(-width_shift_range, width_shift_range)
        if width_shift_range < 1:
            ty *= img.shape[img_col_axis]
    else:
        ty = 0

    if shear_range:
        shear = np.deg2rad(np.random.uniform(-shear_range, shear_range))
    else:
        shear = 0

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

    transform_matrix = None
    if theta != 0:
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                [0, np.cos(shear), 0],
                                [0, 0, 1]])
        transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    perspective_matrix = None
    if perspective_range:
        pts1 = np.float32([[25,25], [25,img.shape[1] - 25], [img.shape[0] - 25,25], [img.shape[0] - 25,img.shape[1] - 25]])
        pts2 = pts1 + (np.random.rand(4,2).astype(np.float32) - .5) * perspective_range * 2
        perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)

    return transform_matrix, perspective_matrix

def apply_random_transform(img, transform_probability, transform_matrix,
                           perspective_probability, perspective_matrix):
    if perspective_probability > .5 and (perspective_matrix is not None):
        img = cv2.warpPerspective(img, perspective_matrix, (img.shape[1], img.shape[0]))
    if transform_probability > .5 and (transform_matrix is not None):
        h, w = img.shape[img_row_axis], img.shape[img_col_axis]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        img = apply_transform(img, transform_matrix, img_channel_axis,
                              fill_mode='constant', cval=0.)

    return img

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

    def load_nuclei(self, image_ids, shape, extra_data_id = None, file_path = TRAIN_PATH_512, alternate_path1 = TRAIN_PATH_CLAHE, alternate_path2 = None):
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
        self.extra_data_id = extra_data_id

        # for now, just mix in extra data as training data
        if extra_data_id:
            image_ids.extend(extra_data_id)

        # Add images
        # Generate specifications of images (i.e. original image sizes
        # and locations). This is more compact than actual images.
        # Images are read in load_image() each time.
        for i, id_ in enumerate(image_ids):
            image_path = file_path + id_ + '/images/' + id_ + '.png'
            alternate_image_path1 = alternate_path1 + id_ + '/images/' + id_ + '.png'
            if alternate_path2:
                alternate_image_path2 = alternate_path2 + id_ + '/images/' + id_ + '.png'
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

# Training dataset
dataset_train = NucleiDataset(augment = True)
dataset_train.load_nuclei(train_ids, config.IMAGE_SHAPE, extra_ids)
dataset_train.prepare()

# Validation dataset
dataset_val = NucleiDataset(augment = False)
dataset_val.load_nuclei(valid_ids, config.IMAGE_SHAPE)
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    #model.load_weights(COCO_MODEL_PATH, stage1_only=True)
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
print('\n\nTraining head\n')
if model.epoch < 1:
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE*2, 
            epochs=1, verbose=2,
            layers='heads')
print('dataset_train.max_masks : ', dataset_train.max_masks)

# Fine tune more layers
print('\n\nTraining 5+')
if model.epoch < 2:
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE*3,
            epochs=2, verbose=2,
            layers="5+") # "5+", "4+", "3+", "2+", "all"

print('\n\nTraining 4+')
if model.epoch < 3:
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE*2,
            epochs=3, verbose=2,
            layers="4+") # "5+", "4+", "3+", "2+", "all"

print('\n\nTraining 3+')
if model.epoch < 4:
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=4, verbose=2,
            layers="3+") # "5+", "4+", "3+", "2+", "all"

print('\n\nTraining 2+')
#for i in range(model.epoch,17):
#    divider = math.ceil((i - 7 + .1)/18)
divider = 1
if model.epoch < 6:
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE/divider,
            epochs=6, verbose=2,
            layers="2+") # "5+", "4+", "3+", "2+", "all"

for i in range(9,29,3):
    model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE/2,
            epochs=i, verbose=2,
            layers="2+") # "5+", "4+", "3+", "2+", "all"
'''
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=10, verbose=2,
            layers="all") # "5+", "4+", "3+", "2+", "all"

model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE/5,
            epochs=25, verbose=2,
            layers="all") # "5+", "4+", "3+", "2+", "all"
'''
'''
class Config512(NucleiConfig):
    IMAGES_PER_GPU = 4
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

config512 = Config512()

# Training dataset
dataset_train = NucleiDataset(augment = True)
dataset_train.load_nuclei(train_ids, config512.IMAGE_SHAPE[0], config512.IMAGE_SHAPE[1], TRAIN_PATH_512)
dataset_train.prepare()

# Validation dataset
dataset_val = NucleiDataset(augment = False)
dataset_val.load_nuclei(valid_ids, config512.IMAGE_SHAPE[0], config512.IMAGE_SHAPE[1], TRAIN_PATH_512)
dataset_val.prepare()

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config512, model_dir=MODEL_DIR)
model.load_weights(model.find_last()[1], by_name=True)
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=30, verbose=2,
            layers="3+") # "5+", "4+", "3+", "all"
'''

class InferenceConfig(NucleiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MAX_INSTANCES = 300
    DETECTION_MIN_CONFIDENCE = 0.90
    IMAGE_RESIZE_MODE = "pad64"

inference_config = InferenceConfig()

# Test dataset
dataset_test = NucleiDataset(augment = False)
dataset_test.load_nuclei(test_ids, config.IMAGE_SHAPE, file_path = TEST_PATH)
dataset_test.prepare()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1] # '/root/briansp/git/Mask_RCNN/logs/nuclei20180322T2343/mask_rcnn_nuclei_0014.h5' # LB .492

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# evaluation on training and validation set
# Compute VOC-Style mAP @ IoU=0.5
'''
APs = []
for image_id in dataset_train.image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_train, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)
print("Training mAP: ", np.mean(APs))
'''
APs = []
for image_id in dataset_val.image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=0.75)
    APs.append(AP)
print("Validation mAP: ", np.mean(APs))

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

new_test_ids = []
rles = []
max_masks = 0
for image_id in dataset_test.image_ids:
    # Load image and ground truth data
    image_info = dataset_test.image_info[image_id]
    #print (image_info['path'])
    image = dataset_test.load_test_image(image_id)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    
    # sort mask from small to large
    idx = list(enumerate(np.sum(r['masks'], axis = (0,1))))
    idx = sorted(idx, key = lambda x: x[1])
    idx = [x for x,y in idx]
    max_masks = max(max_masks, len(idx))
    r['masks'] = r['masks'][:,:,idx]

    #print (image_info['id_'], image.shape, r['masks'].shape, image.shape[0]*image.shape[1])
    # remove duplicate pixels
    mask = np.expand_dims(np.ones(image.shape[:2]), axis=2)
    for i in range(r['masks'].shape[2]):
        r['masks'][:,:,i:i+1] = (mask>.5) * r['masks'][:,:,i:i+1]
        mask[r['masks'][:,:,i:i+1] > .5] = 0

    # Show the test image
    #visualize.display_top_masks(image, resized_mask[:,:,0:1], r['class_ids'][0:1], dataset_train.class_names, limit=1)
    #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    #                            dataset_val.class_names, r['scores'], ax=get_ax())

    # prepare data to make df
    rle = []
    rle = list(mask_to_rles(r['masks']))
    while [] in rle:
        print ("Removing empty mask for ", image_info['id_'])
        rle.remove([])
    rles.extend(rle)
    new_test_ids.extend([image_info['id_']] * len(rle))

print ("Max_masks : ", max_masks)

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
submit_filename = 'sub-retry-2-1.csv'
sub.to_csv(submit_filename, index=False)
print ('Created ', submit_filename)
print ("Done predicting nuclei")
