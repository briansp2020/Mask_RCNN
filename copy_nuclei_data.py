# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import warnings
#import scipy.ndimage as ndi
from multiprocessing import Pool, cpu_count
from functools import partial

from shutil import copyfile
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
#from skimage.io import imread, imsave
#from skimage.transform import resize
#from skimage.color import rgb2gray, gray2rgb

INPUT_DIR  = '/root/Downloads/nuclei'
OUTPUT_DIR = '/root/input'
TRAIN_DIR = 'stage1_train'
TEST_DIR  = 'stage1_test'
#FIXED_DATA_DIR = '/home/briansp/git/kaggle-dsbowl-2018-dataset-fixes/stage1_train'
FIXED_DATA_DIR = None

TRAIN_PATH = os.path.join(INPUT_DIR, TRAIN_DIR)
TRAIN_OUTPUT_384_DIR = os.path.join(OUTPUT_DIR, TRAIN_DIR+'_384')
TRAIN_OUTPUT_512_DIR   = os.path.join(OUTPUT_DIR, TRAIN_DIR+'_512')
TRAIN_OUTPUT_CLAHE_DIR = os.path.join(OUTPUT_DIR, TRAIN_DIR+'_CLAHE')
if not os.path.exists(TRAIN_OUTPUT_512_DIR):
    os.mkdir(TRAIN_OUTPUT_512_DIR)
if not os.path.exists(TRAIN_OUTPUT_384_DIR):
    os.mkdir(TRAIN_OUTPUT_384_DIR)
TEST_PATH  = os.path.join(INPUT_DIR, TEST_DIR)
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, TEST_DIR)
if not os.path.exists(TEST_OUTPUT_DIR):
    os.mkdir(TEST_OUTPUT_DIR)

def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def process_image(image_id, source_dir, fixed_dir, destination_dir, black_and_white, resize, img_height, img_width):
    crop_mask = False
    resize_mask = False
    image_path = os.path.join(source_dir, image_id, 'images', '{}.png'.format(image_id))
    # check if fixed image exists
    if fixed_dir:
        fixed_path = os.path.join(fixed_dir, image_id, 'images', '{}.png'.format(image_id))
        if os.path.exists(fixed_path):
            image_path = fixed_path
            source_dir = fixed_dir 
        else:
            print('Fixed image for {} does not exist'.format(image_id))
    
    image_dest = os.path.join(destination_dir, image_id, 'images')
    if not os.path.exists(image_dest):
        os.makedirs(image_dest)
    image_dest = os.path.join(destination_dir, image_id, 'images', '{}.png'.format(image_id))
    if not black_and_white and not resize:
        copyfile(image_path, image_dest)
    else:
        img = cv2.imread(image_path)

        cmin = 0
        cmax = img.shape[0] - 1
        rmin = 0
        rmax = img.shape[1] - 1
        if resize:
            # check bounding box to see if the image needs to be cropped
            rmin, rmax, cmin, cmax = bbox2(img[:,:,:3])
            if (rmax - rmin + 1) < (img.shape[0] * .9) or \
               (cmax - cmin + 1) < (img.shape[1] * .9):
                img = img[rmin:rmax+1, cmin:cmax+1, :]
                crop_mask = True
           
            if img.shape[0] < img_height and img.shape[1] < img_width:
                img = cv2.resize(img, (img_height, img_width))
                resize_mask = True

        if black_and_white:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0) #,tileGridSize=(gridsize,gridsize))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            #img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        #img = rgb2gray(img)
        #img = gray2rgb(img)
        cv2.imwrite(image_dest, img)

    mask_path = os.path.join(source_dir, image_id, 'masks')
    if os.path.exists(mask_path): # test data does not have masks
        mask_dest = os.path.join(destination_dir, image_id, 'masks')
        if not os.path.exists(mask_dest):
            os.makedirs(mask_dest)
        mask_files = next(os.walk(mask_path))[2]
        for mask in mask_files:
            mask_filepath = os.path.join(mask_path, mask)
            mask_destpath = os.path.join(mask_dest , mask)
            
            if crop_mask or resize_mask:
                mask_img = cv2.imread(mask_filepath)
                #mask_img = ndi.binary_fill_holes(mask_img)
                if crop_mask:
                    mask_img = mask_img[rmin:rmax+1, cmin:cmax+1, :]
                if resize_mask:
                    mask_img = cv2.resize(mask_img, (img_height, img_width))
                cv2.imwrite(mask_destpath, mask_img)
            else:
                copyfile(mask_filepath, mask_destpath)
        
def copy_data(source_dir, fixed_dir, destination_dir, black_and_white = False, resize = False, img_height = 512, img_width = 512):
    # copy data while patching holes in mask
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_ids = next(os.walk(source_dir))[1]
        #process_image('1ec74a26e772966df764e063f1391109a60d803cff9d15680093641ed691bf72',  source_dir=source_dir, destination_dir=destination_dir,
        #                                                        black_and_white=black_and_white, resize = resize)
        p = Pool(cpu_count())
        with tqdm(total = len(data_ids)) as pbar:
            for n, _ in tqdm(enumerate(p.imap_unordered(partial(process_image, source_dir=source_dir, fixed_dir = fixed_dir,
                                                                destination_dir=destination_dir, black_and_white=black_and_white,
                                                                resize = resize, img_height = img_height, img_width = img_width), data_ids))):
                pbar.update()

def generate_test_csv(test_file_dir, output_csv):
    test_df = pd.DataFrame(columns=['ImageId', 'size', 'mult_factor'])
    test_files = next(os.walk(test_file_dir))[1]
    for i, id_ in enumerate(test_files):
        test_file_path = os.path.join(test_file_dir, '%s/images/%s.png'%(id_, id_))
        #print (test_file_path)
        mask = cv2.imread(test_file_path)
        test_df.loc[i] = [id_, mask.shape[:2], 1]

    #print(output_csv)
    test_df.head()
    test_df.to_csv(output_csv)
    
    
if __name__ == '__main__':
    # take care of CSV files first
    #generate_test_csv(os.path.join(INPUT_DIR, 'stage1_test'), os.path.join(OUTPUT_DIR, 'stage1_test_data.csv'))
    copyfile(INPUT_DIR + '/stage1_test_data.csv', OUTPUT_DIR + '/stage1_test_data.csv')
    copyfile(INPUT_DIR + '/stage1_train_data.csv', OUTPUT_DIR + '/stage1_train_data.csv')
    copyfile(INPUT_DIR + '/stage1_val_data.csv', OUTPUT_DIR + '/stage1_val_data.csv')
    
    print ('Processing training images')
    copy_data(TRAIN_PATH, FIXED_DATA_DIR, TRAIN_OUTPUT_512_DIR, black_and_white = False, resize = True, img_height = 512, img_width = 512)
    copy_data(TRAIN_PATH, FIXED_DATA_DIR, TRAIN_OUTPUT_CLAHE_DIR, black_and_white = True, resize = True, img_height = 512, img_width = 512)
    copy_data(TRAIN_PATH, FIXED_DATA_DIR, TRAIN_OUTPUT_384_DIR, black_and_white = False, resize = True, img_height = 384, img_width = 384)
    print ('Processing test images')
    copy_data(TEST_PATH, None, TEST_OUTPUT_DIR, black_and_white = False)

