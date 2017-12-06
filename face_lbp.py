import sys
import os

import glob
import numpy as np
import scipy
import dlib
import sklearn
from skimage import io
from skimage import transform
from skimage import color
from random import randint

import lbp
import cv2
#from imutils.face_utils import FaceAligner
#from imutils.face_utils import rect_to_bb
#import imutils
import face_aligment
import helpers

IMAGE_SIZE = 120
#win = dlib.image_window()
#descriptor = lbp.LBP(24, 3)
descriptor = lbp.LBP(8, 2)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def extract_face(img, is_train):
    #dets = detector(img, 1)
    #img = np.uint8(img)
    #print img.shape
    if len(img.shape) == 2:
        gray = img
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print 'zzz'

    fa = face_aligment.FaceAligner(predictor, desiredFaceWidth=120, desiredFaceHeight=120)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        #print('1st No catch')
        rects = detector(gray, 2)
        if len(rects) == 0:
            if is_train:
                return (np.zeros(8496), np.zeros(8496), False)
            #print('2nd No catch, get Middle')
            height = img.shape[0]
            width = img.shape[1]
            crop_img = img[height/4: height*3/4, width/4: width*3/4]
            #resized_img = transform.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
        else:
            for rect in rects:
                (x, y, w, h) = helpers.rect_to_bb(rect)
                #resized_img = fa.align(img, gray, rect)
                crop_img = fa.align(img, gray, rect)
                break
    else:
        for rect in rects:
            (x, y, w, h) = helpers.rect_to_bb(rect)
            #resized_img = fa.align(img, gray, rect)
            crop_img = fa.align(img, gray, rect)
            break
            #print("Number of faces detected: {}".format(len(dets)))


    # Do LBP

    crop_img = np.uint8(crop_img)
    if len(crop_img.shape) == 2:
        crop_equ = crop_img
    elif len(crop_img.shape) == 3:
        crop_equ = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        print('zzz')


    '''win.clear_overlay()
    win.set_image(img)
    #win.add_overlay(dets)
    dlib.hit_enter_to_continue()
    win.set_image(crop_img)
    dlib.hit_enter_to_continue()'''


    crop_equ = cv2.equalizeHist(crop_equ)
    noise_img = crop_equ + np.random.randint(-2, 3, size=crop_equ.shape)
    #noise_img = np.uint8(noise_img)
    #return (crop_equ.ravel(), noise_img.ravel(), True)
    #return resized_equ.ravel()
    hist = descriptor.hist(crop_equ)
    hist_noise = descriptor.hist(noise_img)
    return (hist, hist_noise, True)


if __name__ == '__main__':

    data_path = 'FinalProject_dataset'
    #np.random.seed(40)

    per_class_train_num = 950
    per_class_val_num = 240

    # Images
    #train_images = np.zeros(shape=[per_class_train_num*8, IMAGE_SIZE*IMAGE_SIZE]) 
    #val_images = np.zeros(shape=[per_class_val_num*8, IMAGE_SIZE*IMAGE_SIZE])
    train_images = np.zeros(shape=[per_class_train_num*8, 8496]) 
    val_images = np.zeros(shape=[per_class_val_num*8, 8496])

    #test_images = np.zeros(shape=[per_class_test_num*8, IMAGE_SIZE*IMAGE_SIZE])
    #test_images = np.zeros(shape=[per_class_test_num*8, 59])


    train_target_classes = np.zeros(per_class_train_num*8 )
    val_target_classes = np.zeros(per_class_val_num*8)

    train_target_classes[per_class_train_num: per_class_train_num*2] = 4
    train_target_classes[per_class_train_num*2: per_class_train_num*3] = 1
    train_target_classes[per_class_train_num*3: per_class_train_num*4] = 5
    train_target_classes[per_class_train_num*4: per_class_train_num*5] = 2
    train_target_classes[per_class_train_num*5: per_class_train_num*6] = 6
    train_target_classes[per_class_train_num*6: per_class_train_num*7] = 3
    train_target_classes[per_class_train_num*7: per_class_train_num*8] = 7

    val_target_classes[per_class_val_num: per_class_val_num*2] = 4
    val_target_classes[per_class_val_num*2: per_class_val_num*3] = 1
    val_target_classes[per_class_val_num*3: per_class_val_num*4] = 5
    val_target_classes[per_class_val_num*4: per_class_val_num*5] = 2
    val_target_classes[per_class_val_num*5: per_class_val_num*6] = 6
    val_target_classes[per_class_val_num*6: per_class_val_num*7] = 3
    val_target_classes[per_class_val_num*7: per_class_val_num*8] = 7


    train_id = 0
    val_id = 0
    #test_id = 0
    is_train = False
    for age in ('child', 'young', 'adult', 'elder'):
        class_path = os.path.join(data_path, age)
        for gender in ('male', 'female'):
            class_gender_path = os.path.join(class_path, gender)
            file_num = len(glob.glob(os.path.join(class_gender_path, '*')))
            #print file_num
            all_pic = np.arange(file_num)
            np.random.shuffle(all_pic)
            train_num = file_num * 0.8
            val_num = file_num * 0.2
            #test_num = file_num * 0.15
            print 'starting' + class_gender_path +': '
            print 'reading training examples'
            train_samples = 0
            while train_samples < per_class_train_num / 2:
                train_pic_id = randint(0, int(train_num) - 1)
                img = io.imread(os.path.join(class_gender_path, str(all_pic[train_pic_id]) + '.jpg'))
                print(os.path.join(class_gender_path, str(all_pic[train_pic_id]) + '.jpg'))
                #train_images[train_id] = extract_face(os.path.join(class_gender_path, str(all_pic[train_pic_id])+'.jpg'))
                is_train = True
                train_images[train_id], train_images[train_id+1], is_find = extract_face(img, is_train)
                if not is_find:
                    continue
                train_id += 2
                train_samples += 1
            print 'reading validation examples'
            val_samples = 0
            while val_samples < per_class_val_num:
                val_pic_id = randint(int(train_num), int(train_num+val_num) - 1)
                #print(os.path.join(class_gender_path, str(all_pic[train_pic_id]) + '.jpg'))
                is_train = False
                img = io.imread(os.path.join(class_gender_path, str(all_pic[val_pic_id]) + '.jpg'))
                val_images[val_id], _, _ = extract_face(img, is_train)
                val_id += 1
                val_samples += 1


    np.save('train_images', train_images)
    np.save('train_targets', train_target_classes)

    np.save('val_images', val_images)
    np.save('val_targets', val_target_classes)

