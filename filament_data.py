from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras import backend as K


obj1 = [128,128,128]
obj2 = [128,0,0]

COLOR_DICT = np.array([obj1, obj2])


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255.0
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255.0
        mask = mask /255.0
        mask[mask > 0.19] = 1
        mask[mask <= 0.19] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = 'data/filament/train/test/',target_size = (512,512),seed = 1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def validationGenerator(batch_size,valid_path,image_folder,mask_folder,image_color_mode = "grayscale",mask_color_mode = "grayscale",flag_multi_class = False,num_class = 2,target_size = (512,512),seed = 1):
    validimg_datagen = ImageDataGenerator(rescale=1. / 255)
    validmask_datagen= ImageDataGenerator(rescale=1. / 255)
    validimg_generator= validimg_datagen.flow_from_directory(
        valid_path,
        classes = [image_folder],
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        seed=seed)

    validmask_generator= validmask_datagen.flow_from_directory(
        valid_path,
        classes = [mask_folder],
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        seed=seed)

    valid_generator = zip(validimg_generator, validmask_generator)
    for (img1,mask1) in valid_generator:
        img1,mask1 = adjustData(img1,mask1,flag_multi_class,num_class)
        yield (img1,mask1)

def testGenerator(test_path,target_size = (512,512),flag_multi_class = False,as_gray = True):
    for file in os.listdir(test_path):
            if (os.path.splitext(file)[-1][1:] == "jpg"):
                print(file)
                img = io.imread(os.path.join(test_path,file),as_gray = as_gray)
                img = img / 255.0
                img = trans.resize(img,target_size)
                img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
                img = np.reshape(img,(1,)+img.shape)
                yield img



def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.jpg"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255.0



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
     for i,item in enumerate(npyfile):
         img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
         io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)






