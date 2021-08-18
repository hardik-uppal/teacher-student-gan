import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl

import data
import module

# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--experiment_dir', default ='/home/Face_rec/CurtinFaces')

py.arg('--batch_size', type=int, default=32)
test_args = py.args()
#
args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))

args.__dict__.update(test_args.__dict__)
args.datasets_dir = '/home/Face_rec/CurtinFaces_crop/'
#'/home/harry/Face_rec/IIITD/fold5'

# ==============================================================================
# =                                    test                                    =
# ==============================================================================


################ CurtinFaces ################

RGB_T_image_dir = [py.join(args.datasets_dir,'RGB','train'), 
                   py.join(args.datasets_dir,'RGB','test')]
A_T_img_paths, A_T_img_paths_test = data.path_extarct(RGB_T_image_dir, args.dataset_target)
A_T_dataset_test = data.make_dataset(A_T_img_paths, args.batch_size, args.load_size, args.crop_size,
                                            training=False, drop_remainder=False, shuffle=False, repeat=1)



# model
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

# resotre
tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join(args.experiment_dir, 'checkpoints')).restore()


@tf.function
def sample_A2B(A):
    A2B = G_A2B(A, training=False)
    A2B2A = G_B2A(A2B, training=False)
    return A2B, A2B2A


@tf.function
def sample_B2A(B):
    B2A = G_B2A(B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return B2A, B2A2B



i = 0
for A in A_T_dataset_test:
    A2B, A2B2A = sample_A2B(A)
    for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
#        img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B2A_i.numpy()], axis=1)
        img = A2B_i.numpy()
        save_dir = py.split(A_T_img_paths[i].replace('RGB','est_depth'))[0]
        py.mkdir(save_dir)
        im.imwrite(img, py.join(save_dir, py.name_ext(A_T_img_paths[i])))
        i += 1

