import numpy as np
import tensorflow as tf
import tf2lib as tl
import os

def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize(img, [crop_size, crop_size])  # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size, training, shuffle=True, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset

def make_zip_dataset_SemiSupervised(A_img_paths, B_img_paths, A_T_img_paths, batch_size, load_size, crop_size, training, shuffle=False, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = A_T_img_paths = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        elif len(A_img_paths) <= len(B_img_paths):
            A_repeat = None  # cycle the shorter one
            B_repeat = 1
        if len(A_img_paths) >= len(A_T_img_paths):
            A_repeat = 1  
            A_T_repeat = None # cycle the shorter one          
        else:
            A_repeat = None  # cycle the shorter one
            A_T_repeat = 1
    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=B_repeat)
    A_T_dataset = make_dataset(A_T_img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=shuffle, repeat=A_T_repeat)

    A_B_AT_dataset = tf.data.Dataset.zip((A_dataset, B_dataset, A_T_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths), len(A_T_img_paths)) // batch_size

    return A_B_AT_dataset, len_dataset

class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)

def path_extarct(image_dir, dataset, test_train=None):
    image_full_arr = []
    label_full_arr = []
#        self.batch_size = batch_size

    #check if input dir is list or not; create two lsit with addresses and corresponding labels
    if isinstance(image_dir, list):
        for image_dir_temp in image_dir:
            label_list = os.listdir(image_dir_temp)
            
            for label in label_list:
                label_dir = os.path.join(image_dir_temp,label)
                for image in os.listdir(label_dir):
                    if image[-3:] in ['jpg','bmp','png']:
                        image_path = os.path.join(label_dir,image)
                        image_full_arr.append(image_path)
                        label_full_arr.append(label)
                    
    else:
        label_list = os.listdir(image_dir)
        
        
        for label in label_list:
            label_dir = os.path.join(image_dir,label)
            for image in os.listdir(label_dir):
                if image[-3:] in ['jpg','bmp','png']:
                    image_path = os.path.join(label_dir,image)
                    image_full_arr.append(image_path)
                    label_full_arr.append(label)
    
    
    if test_train:
        ## join both list for filtering
        arr_joined = np.column_stack((image_full_arr,label_full_arr))
        # if test_train exists, set the filter for test labels
        if dataset == 'pandora':
            filter = np.asarray(['10.0','14.0','16.0','20.0'])
        elif dataset == 'IIITD':
            filter = np.asarray(['10.0','14.0','16.0','20.0'])
        elif dataset == 'CurtinFaces':
            filter = np.asarray(['10','14','16','20'])
        elif dataset == 'Lock3d':
            label_vals = os.listdir(label_dir)
            
            filter = np.asarray(['674', '675', '676', '677', '678', '679', '680', '681', '682', '683'
                                 '764', '765', '766', '767', '768', '769', '770', '771', '772', '773', '999'])
        elif dataset == 'EURECOM_Kinect_db':
            filter = np.asarray(['0010','0015','0020','0025','0030','0035','0040','0041','0042','0043','0044','0045','0050','0051','0052'])
        

        test_arr = arr_joined[np.in1d(arr_joined[:, 1], filter)]
        train_arr = arr_joined[np.in1d(arr_joined[:, 1], filter, invert=True)]
        
        image_train_arr = train_arr[:,0]
        label_train_arr = train_arr[:,1].astype(np.float)
        image_test_arr = test_arr[:,0]
        label_test_arr = test_arr[:,1].astype(np.float)
        

    else:
        image_train_arr = image_full_arr
        image_test_arr = None
 

#        num_classes_full = len(set(label_full_arr))
    return image_train_arr, image_test_arr
