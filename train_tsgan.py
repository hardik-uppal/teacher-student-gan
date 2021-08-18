import functools

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm

import data
import module

#TODO
#ADD reverse hubber loss
# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--dataset', default='CurtinFaces')
py.arg('--dataset_target', default='imdb_face')
py.arg('--datasets_dir', default='/home/harry/Face_rec/CurtinFaces_crop/')
py.arg('--datasets_dir_target', default='/home/harry/Face_rec/imbdface/')
py.arg('--load_size', type=int, default=128)  # load image to this size
py.arg('--crop_size', type=int, default=128)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=100)
py.arg('--epoch_decay_teach', type=int, default=10)  # epoch to start decaying learning rate
py.arg('--epoch_decay', type=int, default=20)
py.arg('--lr', type=float, default=0.000002)
py.arg('--lr_teach', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.9)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--MAE_teacher_loss_weight', type=float, default=10)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()


#args.dataset = 'rgb'
# output_dir
output_dir = py.join(args.dataset_target+'_depth_est_imdb_9jan', args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


# ==============================================================================
# =                                    data                                    =
# ==============================================================================

#A_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainA'), '*.jpg')
#B_img_paths = py.glob(py.join(args.datasets_dir, args.dataset, 'trainB'), '*.jpg')
RGB_image_dir = [py.join(args.datasets_dir, 'RGB','train'),py.join(args.datasets_dir, 'RGB','test')]
A_img_paths, A_img_paths_test = data.path_extarct(RGB_image_dir, args.dataset)


Depth_image_dir = [py.join(args.datasets_dir, 'depth','train'),py.join(args.datasets_dir, 'depth','test')]
B_img_paths, B_img_paths_test = data.path_extarct(Depth_image_dir, args.dataset)

##lock3dface
#RGB_T_image_dir = [py.join(args.datasets_dir_target, 'RGB','train'),py.join(args.datasets_dir_target, 'RGB','test_ps'),py.join(args.datasets_dir_target, 'RGB','test_oc'),py.join(args.datasets_dir_target, 'RGB','test_fe')
#                    ,py.join(args.datasets_dir_target, 'RGB','test_2')]

##for eurecom, lfw
RGB_T_image_dir = [py.join(args.datasets_dir_target, 'IMDB_new')]
#A_T_img_paths, A_T_img_paths_test = data.path_extarct(RGB_T_image_dir, args.dataset_target)

##for IIITD
#RGB_T_image_dir = [py.join(args.datasets_dir_target,'train','RGB'), 
#                   py.join(args.datasets_dir_target,'test','RGB')]



A_T_img_paths, A_T_img_paths_test = data.path_extarct(RGB_T_image_dir, args.dataset_target)

A_B_AT_dataset, len_dataset = data.make_zip_dataset_SemiSupervised(A_img_paths, B_img_paths, A_T_img_paths, args.batch_size, args.load_size, args.crop_size, training=True, shuffle=False, repeat=False)

A_S2B_S_pool = data.ItemPool(args.pool_size)
A_T2B_T_pool = data.ItemPool(args.pool_size)
B_T2A_T_pool = data.ItemPool(args.pool_size)

#A_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testA'), '*.jpg')
#B_img_paths_test = py.glob(py.join(args.datasets_dir, args.dataset, 'testB'), '*.jpg')
A_T_dataset_test, _ = data.make_zip_dataset(A_T_img_paths, A_T_img_paths, args.batch_size, args.load_size, args.crop_size, training=False, repeat=True)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================
## Generator for Student
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

## Generator for Teacher
#G_A2B_teacher = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
#G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
mae_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_lr_scheduler_teach = module.LinearDecay(args.lr_teach, args.epochs * len_dataset, args.epoch_decay_teach * len_dataset)

D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay_teach * len_dataset)

G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
G_optimizer_teach = keras.optimizers.Adam(learning_rate=G_lr_scheduler_teach, beta_1=args.beta_1)

D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================
  
@tf.function
def train_G_teacher(A_S, B_S):
    with tf.GradientTape() as t:
        #orginal teacher cycle
        A_S2B_S = G_A2B(A_S, training=True)

        A_S2B_S_d_logits = D_B(A_S2B_S, training=True)

        A_S2B_S_g_loss = g_loss_fn(A_S2B_S_d_logits)
        #MAE loss for teacher
        A2B_B_S_mse_loss = mae_loss_fn(A_S2B_S, B_S)


        G_loss = (A_S2B_S_g_loss) + (A2B_B_S_mse_loss) * args.MAE_teacher_loss_weight
#                 + B2A2B_cycle_loss) 
#        * args.cycle_loss_weight + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables)
#    + G_B2A.trainable_variables)
    G_optimizer_teach.apply_gradients(zip(G_grad, G_A2B.trainable_variables))
#                                    + G_B2A.trainable_variables))

    return A_S2B_S, {'A_S2B_S_g_loss': A_S2B_S_g_loss}
#                      'B2A_g_loss': B2A_g_loss,
#                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
#                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
#                      'A2A_id_loss': A2A_id_loss,
#                      'B2B_id_loss': B2B_id_loss}
  
@tf.function
def train_G(A_T):
    with tf.GradientTape() as t:

        #student
        A_T2B_T = G_A2B(A_T, training=True)
#        B2A_T = G_B2A(A2B_T, training=True)
        A_T2B_T2A_T = G_B2A(A_T2B_T, training=True)
#        B2A2B = G_A2B(B2A, training=True)
#        A2A = G_B2A(A, training=True)
#        B2B = G_A2B(B, training=True)

        A_T2B_T_d_logits = D_B(A_T2B_T, training=True)
        B_T2A_T_d_logits = D_A(A_T2B_T2A_T, training=True)

        A_T2B_T_g_loss = g_loss_fn(A_T2B_T_d_logits)
        B_T2A_T_g_loss = g_loss_fn(B_T2A_T_d_logits)
        A_T2B_T2A_T_cycle_loss = cycle_loss_fn(A_T, A_T2B_T2A_T)
#        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
#        A2A_id_loss = identity_loss_fn(A, A2A)
#        B2B_id_loss = identity_loss_fn(B, B2B)

        G_loss = (A_T2B_T_g_loss + B_T2A_T_g_loss) + (A_T2B_T2A_T_cycle_loss) * args.cycle_loss_weight 
#                 + B2A2B_cycle_loss) * args.cycle_loss_weight 
#        + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A_T2B_T, A_T2B_T2A_T, {'A_T2B_T_g_loss': A_T2B_T_g_loss,
                      'B_T2A_T_g_loss': B_T2A_T_g_loss,
                      'A_T2B_T2A_T_cycle_loss': A_T2B_T2A_T_cycle_loss}
#                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
#                      'A2A_id_loss': A2A_id_loss,
#                      'B2B_id_loss': B2B_id_loss}


@tf.function
def train_D(A_S, B_S, A_S2B_S, A_T2B_T, A_T2B_T2A_T):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A_S, training=True)
        B2A_d_logits = D_A(A_T2B_T2A_T, training=True)
        #train for teacher
        B_d_logits = D_B(B_S, training=True)
        A2B_d_logits = D_B(A_S2B_S, training=True)
        #train for student
        B_d_logits_T = D_B(B_S, training=True)
        A2B_d_logits_T = D_B(A_T2B_T, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        B_d_loss_T, A2B_d_loss_T = d_loss_fn(B_d_logits_T, A2B_d_logits_T)
        D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A_S, A_T2B_T2A_T, mode=args.gradient_penalty_mode)
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B_S, A_S2B_S, mode=args.gradient_penalty_mode)
        D_B_gp_T = gan.gradient_penalty(functools.partial(D_B, training=True), B_S, A_T2B_T, mode=args.gradient_penalty_mode)

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (B_d_loss_T + A2B_d_loss_T) + (D_A_gp + D_B_gp + D_B_gp_T) * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'B_d_loss_T': B_d_loss_T + A2B_d_loss_T,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp,
            'D_B_gp_T': D_B_gp_T}


def train_step(A_S, B_S, A_T):
    #train teacher
    A_S2B_S, G_loss_dict_S = train_G_teacher(A_S, B_S)
    A_T2B_T, A_T2B_T2A_T, G_loss_dict_T = train_G(A_T)
    G_loss_dict = dict(G_loss_dict_S, **G_loss_dict_T)
    # cannot autograph `A2B_pool`
    A_S2B_S = A_S2B_S_pool(A_S2B_S)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    A_T2B_T = A_T2B_T_pool(A_T2B_T)
    A_T2B_T2A_T = B_T2A_T_pool(A_T2B_T2A_T)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A_S, B_S, A_S2B_S, A_T2B_T, A_T2B_T2A_T)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A_S, A_T):
    A_S2B_S = G_A2B(A_S, training=False)
    A_T2B_T = G_A2B(A_T, training=False)
#    B_S2A_S = G_B2A(B_S, training=False)#didn't train
    
#    A_S2B_S2A_S = G_B2A(A_S2B_S, training=False)
    A_T2B_T2A_T = G_B2A(A_T2B_T, training=False)

    return A_S2B_S, A_T2B_T, A_T2B_T2A_T
#                A_S2B_S2A_S, B_S2A_S,



# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_B2A=G_B2A,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                G_optimizer_teach=G_optimizer_teach,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'),
                           max_to_keep=5)

    
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_T_dataset_test)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

# main loop
with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)

        # train for an epoch
        for A_S, B_S, A_T in tqdm.tqdm(A_B_AT_dataset, desc='Inner Epoch Loop', total=len_dataset):
            G_loss_dict, D_loss_dict = train_step(A_S, B_S, A_T)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')
            
#            img = im.immerge(np.concatenate([A_S, B_S, A_T], axis=0), n_rows=1)
#            im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

            # sample
            if G_optimizer.iterations.numpy() % 100 == 0:
                A_T, _ = next(test_iter)
                A_S2B_S, A_T2B_T, A_T2B_T2A_T = sample(A_S, A_T)
                img = im.immerge(np.concatenate([A_S, B_S, A_S2B_S, A_T, A_T2B_T, A_T2B_T2A_T], axis=0), n_rows=2)
                im.imwrite(img, py.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()))

        # save checkpoint
        checkpoint.save(ep)
