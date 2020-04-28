import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import tensorflow.contrib.slim as slim
import os
import os.path
import cv2
import time

import utils
import network

Flags = tf.flags.FLAGS
tf.flags.DEFINE_string('f', '', 'kernel')

# Optimization Flags
tf.flags.DEFINE_integer('batch_train', 10, 'number of images in one training batch')
tf.flags.DEFINE_integer('batch_test', 8, 'number of images in one testing batch')
tf.flags.DEFINE_integer('max_step', 2000, 'max steps to train')
tf.flags.DEFINE_integer('img_height', 224, 'size of height of input images')
tf.flags.DEFINE_integer('img_width', 224, 'size of width of input images')
tf.flags.DEFINE_float('init_learning_rate', 0.0001, 'initial learning rate for adam optimizer')
tf.flags.DEFINE_string('log_dir', './logs/', 'path to store training parameters')

# Dataset Flags
tf.flags.DEFINE_string('gt_dir', './data/GT_512_128/', 'path to ground truth')
tf.flags.DEFINE_string('data_ft_dir', './data/flatten/', 'path to pre-processed data')
tf.flags.DEFINE_integer('input_channel', 3, 'number of channels in one input image')
tf.flags.DEFINE_integer('num_classes', 2, 'number of total categorys to classify')
tf.flags.DEFINE_integer('img_num', 128, 'number of B-scans in one 3D cube-scan')

# Fine-Tuning Flags
tf.flags.DEFINE_string('scope', 'vgg_16', 'name of the backbone classification network')
tf.flags.DEFINE_bool('load_ckpt', True, 'whether to load pre-train parameters: True/False')
tf.flags.DEFINE_string('vgg_16_ckpt_dir', './load/vgg_16.ckpt', 'path to load pre-train parameters')
tf.flags.DEFINE_string('resnet_v1_50_ckpt_dir', './load/resnet_v1_50.ckpt', 'path to load pre-train parameters')
tf.flags.DEFINE_string('InceptionV3_ckpt_dir', './load/inception_v3.ckpt', 'path to load pre-train parameters')

scope_list = ['vgg_16','resnet_v1_50','InceptionV3']
assert Flags.scope in scope_list, 'Undefined Backbone'

def prepprocess(image, label):
    label = tf.cast(label, tf.int32)
    image = tf.cast(image, tf.string)
    image = tf.read_file(image)
    image = tf.image.decode_bmp(image, channels=1)
    image = tf.concat([image] * Flags.input_channel, 2)
    image = tf.image.resize_images(image, [Flags.img_height, Flags.img_width])
    image = tf.image.per_image_standardization(image)
    label = tf.one_hot(label, depth=Flags.num_classes)
    return image, label

def train(data_list, label_list, k=-1):
    tf.reset_default_graph()
    image, label = utils.balanced_train_sampling(data_list, label_list)
    num_samples = len(label_list)
    num_epochs = (Flags.max_step * Flags.batch_train) // num_samples + 1
    dataset =tf.data.Dataset.from_tensor_slices((image, label)).map(prepprocess)
    dataset = dataset.repeat(num_epochs).shuffle(num_samples).batch(Flags.batch_train)
    itreator = dataset.make_one_shot_iterator()
    image_batch, label_batch = itreator.get_next()
    slim.create_global_step()
    x = tf.placeholder(tf.float32, shape=[Flags.batch_train, Flags.img_height, Flags.img_width, Flags.input_channel])
    y = tf.placeholder(tf.int16, shape=[Flags.batch_train, Flags.num_classes])

    model = network.Model(imgs=x,
                          num_classes=Flags.num_classes,
                          scope=Flags.scope,
                          img_height=Flags.img_height,
                          img_width=Flags.img_width,
                          is_training=True)

    excitation_list, output = model.build()
    accuracy = utils.accuracy(output, y)
    loss = slim.losses.softmax_cross_entropy(output, y)
    optimizer = tf.train.AdamOptimizer(learning_rate=Flags.init_learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        loss = control_flow_ops.with_dependencies([updates], loss)
    train_op = slim.learning.create_train_op(loss, optimizer)

    saver = tf.train.Saver(tf.global_variables())
    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(init)
    if Flags.load_ckpt:
        print('loading pretrained weights')
        if Flags.scope == 'vgg_16':
            pre_trained_weights = Flags.vgg_16_ckpt_dir
            utils.load_ckpt_with_skip(pre_trained_weights, sess,
                                      skip_layer=['global_step ', 'vgg_16/fc8/biases', 'vgg_16/fc8/weights'])
            print('loading success')
        elif Flags.scope == 'resnet_v1_50':
            pre_trained_weights = Flags.resnet_v1_50_ckpt_dir
            utils.load_ckpt_with_skip(pre_trained_weights, sess,
                                      skip_layer=['global_step ', 'resnet_v1_50/logits/weights',
                                                  'resnet_v1_50/logits/biases'])
            print('loading success')
        elif Flags.scope == 'InceptionV3':
            pre_trained_weights = Flags.InceptionV3_ckpt_dir
            utils.load_ckpt_with_skip(pre_trained_weights, sess,
                                      skip_layer=['global_step ','InceptionV3/AuxLogits/Conv2d_1b_1x1/weights',
                                                  'InceptionV3/AuxLogits/Conv2d_2b_1x1/weights',
                                                  'InceptionV3/Logits/Conv2d_1c_1x1/weights'])

    print('training...')
    start_time = time.process_time()
    for step in np.arange(Flags.max_step):
        tra_images, tra_labels = sess.run([image_batch, label_batch])
        _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy], feed_dict={x: tra_images, y: tra_labels})
        elapsed = round((time.process_time() - start_time) / 60, 2)
        print('Fold: %d, Step: %d, Loss: %.4f, Accuracy: %.4f%%, Time: %.2fmin' % (k + 1, step, tra_loss, tra_acc, elapsed))
        if  (step + 1) == Flags.max_step:
            checkpoint_path = Flags.log_dir + 'fold/{}/'.format(k + 1)
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            saver.save(sess, checkpoint_path + 'mode.ckpt', global_step=step + 1)
    sess.close()

def test(data_list, label_list, log_dir):
    with tf.Graph().as_default():
        n_test = len(label_list)
        n_cube = n_test // Flags.img_num
        num_step = n_test // Flags.batch_test // n_cube

        dataset = tf.data.Dataset.from_tensor_slices((data_list, label_list)).map(prepprocess).batch(Flags.batch_test)
        itreator = dataset.make_one_shot_iterator()
        image_batch, label_batch = itreator.get_next()
        x = tf.placeholder(tf.float32,shape=[Flags.batch_test, Flags.img_height, Flags.img_width, Flags.input_channel])
        y = tf.placeholder(tf.int16, shape=[Flags.batch_test, Flags.num_classes])

        model = network.Model(imgs=x,
                              num_classes=Flags.num_classes,
                              scope=Flags.scope,
                              img_height=Flags.img_height,
                              img_width=Flags.img_width,
                              is_training=False)
        excitation_list, output = model.build()
        correct = utils.num_correct_prediction(output, y)
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            class_correct = 0
            for i in range(n_cube):
                start_time = time.process_time()
                cube_name = data_list[i * Flags.img_num].split('/')[-2]
                print(cube_name)
                cube_correct = 0
                for step in range(num_step):
                    tes_images, tes_labels = sess.run([image_batch, label_batch])
                    outs_train, batch_correct = sess.run([output, correct],feed_dict={x:tes_images, y:tes_labels})
                    cube_correct += np.sum(batch_correct)
                    class_correct += np.sum(batch_correct)
                cube_correct = cube_correct / Flags.img_num
                print('Total accuracy: %.2f%%' % (100 * cube_correct))
                elapsed = round((time.process_time() - start_time), 2)
                print('time used', elapsed)
            class_correct = class_correct / n_test
            print('Total accuracy: %.2f%%' % (100 * class_correct))

            sess.close()
            return(class_correct)

def corss_val_5():
    class_correct = []
    for i in range(5):
        data_train_list = []
        label_train_list = []
        data_test_list = []
        label_test_list = []
        for j in range(5):
            gt_dir = Flags.gt_dir + 'Kfold/{}'.format(j + 1)
            filenames = [filename for filename in os.listdir(gt_dir) if filename.endswith('.bmp')]
            for file in filenames:
                im = cv2.imread(gt_dir + '/' + file, 0)
                # Add the B-scans and add the category label.
                if j == i:
                    for k in range(Flags.img_num):
                        data_test_list.append(Flags.data_ft_dir + file.replace('.bmp', '_VBM4D/{}_Flatten.bmp'.format(k + 1)))
                        label_test_list.append(int(np.sum(im[:, k]) > 0))
                else:
                    for k in range(Flags.img_num):
                        data_train_list.append(Flags.data_ft_dir + file.replace('.bmp', '_VBM4D/{}_Flatten.bmp'.format(k + 1)))
                        label_train_list.append(int(np.sum(im[:, k]) > 0))
        train(data_train_list, label_train_list, i)
        log_dir = Flags.log_dir + 'fold/{}/'.format(i + 1)
        k_correct = test(data_test_list, label_test_list, log_dir)
        class_correct.append(k_correct)
    print(class_correct)
    print('mean:', sum(class_correct) / len(class_correct))

if __name__ == '__main__':
    # Choose which GPU or CPU to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    corss_val_5()







