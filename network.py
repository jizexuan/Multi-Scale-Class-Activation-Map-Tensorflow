import tensorflow as tf
from tensorflow.contrib.slim import nets
import tensorflow.contrib.slim as slim

class Model():
    """Construct a multi-scale neural network.
    Images are classified through this multi-scale structure network.
    Generate multi-scale class activation maps (MS-CAM) using the weakly supervised label.
    Integrate the Attentional Fully Connected layer (AFC).
    
    Args:
        imgs: 4D tensor, a batch of input images
        num_classes: Integer, number of total categorys to classify.
        scope: Name of the backbone classification network: 'vgg_16','resnet_v1_50','InceptionV3'.
        img_height: Integer, size of height of input images.
        img_width: Integer, size of width of input images.
        is_training: Bool, whether to load pre-train parameters: True/False
        sus_channels: Integer, number of the output channels each SUS module.
    """
    def __init__(self, imgs, num_classes, scope, img_height, img_width, is_training, sus_channels = 64):
        self.scope = scope
        self.img_height = img_height
        self.img_width = img_width
        self.is_training = is_training
        self.num_classes = num_classes
        if scope == 'vgg_16':
            self.logit, self.end_points = nets.vgg.vgg_16(imgs,
                                                num_classes=num_classes,
                                                is_training=is_training,
                                                dropout_keep_prob=0.5,
                                                spatial_squeeze=True,
                                                scope=scope)

            self.conv_list = ['vgg_16/conv1/conv1_2',
                              'vgg_16/conv2/conv2_2',
                              'vgg_16/conv3/conv3_3',
                              'vgg_16/conv4/conv4_3',
                              'vgg_16/conv5/conv5_3']
        elif scope == 'resnet_v1_50':
            self.logit, self.end_points = nets.resnet_v1.resnet_v1_50(imgs,
                                                            num_classes=num_classes,
                                                            is_training=is_training,
                                                            global_pool=True,
                                                            output_stride=None,
                                                            reuse=None,
                                                            scope=scope)
            self.conv_list = ['resnet_v1_50/conv1',
                              'resnet_v1_50/block1',
                              'resnet_v1_50/block2',
                              'resnet_v1_50/block3',
                              'resnet_v1_50/block4']
        else:
            self.logit, self.end_points = nets.inception.inception_v3(imgs,
                                                            num_classes=num_classes,
                                                            is_training=is_training,
                                                            dropout_keep_prob=0.8,
                                                            min_depth=16,
                                                            depth_multiplier=1.0,
                                                            prediction_fn=slim.softmax,
                                                            spatial_squeeze=True,
                                                            reuse=None,
                                                            scope=scope)
            self.conv_list = ['Conv2d_2b_3x3',
                              'Conv2d_4a_3x3',
                              'Mixed_5d',
                              'Mixed_6e',
                              'Mixed_7c']
        # Extract the required feature graph of backbone
        fmp = []
        for i in range(len(self.conv_list)):
            conv_ = self.end_points[self.conv_list[i]]
            fmp_ = self._sus_conv(conv_, is_training=is_training, scope='conv_{}'.format(i + 1),
                              n_channels=sus_channels)
            fmp.append(fmp_)
        self.fmp = tf.concat(fmp, -1)

    def _spatial_pooling(self, x, k, alpha=None, scope='spatial_pool'):
        # From the project WILDCAT：https://github.com/Irlyue/wildcat
        """
            Operation for spatial pooling.
            :param x: Tensor, with shape(batch_size, h, w, c)
            :param k: int,
            :param alpha: float, mixing coefficient for kmax and kmin. If none, ignore kmin.
            :param scope: str, parameter scope
            :return:
                op: Tensor, with shape(batch_size, c)
            """
        with tf.variable_scope(scope):
            batch_size, _, _, n_classes = x.get_shape().as_list()
            x_flat = tf.reshape(x, shape=(batch_size, -1, n_classes))
            x_transp = tf.transpose(x_flat, perm=(0, 2, 1))
            k_maxs, _ = tf.nn.top_k(x_transp, k, sorted=False)
            k_maxs_mean = tf.reduce_mean(k_maxs, axis=2)
            result = k_maxs_mean
            if alpha:
                # top -x_flat to retrieve the k smallest values
                k_mins, _ = tf.nn.top_k(-x_transp, k, sorted=False)
                # flip back
                k_mins = -k_mins
                k_mins_mean = tf.reduce_mean(k_mins, axis=2)
                alpha = tf.constant(alpha, name='alpha', dtype=tf.float32)
                result += alpha * k_mins_mean
            return result

    def _sus_conv(self, x, is_training, n_channels, scope='conv_multi'):
        with slim.arg_scope([slim.batch_norm], is_training=is_training, decay=0.9):
            # conv 1*1 and BN
            fmp = slim.conv2d(x, n_channels, [1, 1], activation_fn=None, normalizer_fn=slim.batch_norm,
                              scope=scope + '_p')
            # Upsampling to the size of input
            fmp = tf.image.resize_bilinear(fmp, [self.img_height, self.img_width])
            return fmp

    def build(self, reduction_ratio = 0.5, excitation_act = tf.nn.sigmoid):
        """
        Operation for spatial pooling.
        reduction_ratio: The scaling rate of the encoder part.
        excitation_act: The activation function of the decoder phase.
        """
        with slim.arg_scope([slim.conv2d], biases_initializer=None, normalizer_fn=None, activation_fn=None):
            with slim.arg_scope([slim.batch_norm], is_training=self.is_training, decay=0.9):
                in_channel = self.fmp.get_shape().as_list()[-1]
                # global pooling
                x_global = self._spatial_pooling(self.fmp, 256*16, alpha=1)
                excitation_list = []
                output = []
                # perform the AFC
                for i in range(self.num_classes):
                    excitation_ = slim.conv2d(tf.reshape(x_global, [-1, 1, 1, in_channel]),
                                           int(in_channel * reduction_ratio),
                                           [1, 1], scope='encoder_{}'.format(i), activation_fn=tf.nn.relu)
                    excitation_ = slim.conv2d(excitation_, in_channel, [1, 1], scope='decoder_{}'.format(i),
                                              normalizer_fn=None, activation_fn=excitation_act)
                    excitation_list.append(excitation_)
                    excitation_ = tf.squeeze(excitation_, [1, 2])
                    output_ = x_global * excitation_
                    output_ = tf.reduce_mean(output_, -1, keepdims=True)
                    output.append(output_)
                output = tf.concat(output, axis=-1)
                return excitation_list, output

    def get_cam(self, excitation):
        classmap = self.fmp * excitation
        classmap = tf.reduce_mean(classmap, -1, keepdims=True)
        classmap = tf.image.resize_bilinear(classmap, [self.img_height, self.img_width])
        classmap = tf.squeeze(classmap, [-1])
        return classmap

    def get_grad_cam_plusplus(self, output):
        conv_sum = tf.reduce_sum(self.fmp, [1, 2], keepdims=True)
        preact_grads = tf.gradients(output, self.fmp)[0]
        act_grads = tf.gradients(tf.exp(output), self.fmp)[0]
        alpha_kc = 1 / (2 + conv_sum * preact_grads)
        w_kc = alpha_kc * tf.nn.relu(act_grads)
        w_kc = tf.reduce_sum(w_kc, [1, 2], keepdims=True)
        classmap = self.fmp * w_kc
        classmap = tf.reduce_sum(classmap, axis=3)
        return classmap






