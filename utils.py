import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import time
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral,create_pairwise_gaussian

def normalize(img, s=0.1):
    """Normalize the image range for visualization"""
    z = img / np.std(img)
    return np.uint8(np.clip((z - z.mean()) / max(z.std(), 1e-4) * s + 0.5, 0, 1) * 255)

def accuracy(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor,
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)*100.0
        tf.summary.scalar(scope+'/accuracy', accuracy)
    return accuracy

def num_correct_prediction(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Return:
      the number of correct predictions
    """
    correct = tf.equal(tf.math.argmax(logits, 1), tf.math.argmax(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct

def load_ckpt_with_skip(data_path, session, skip_layer=[]):
    """Load the pre-training parameters"""
    s_time = time.clock()
    reader = pywrap_tensorflow.NewCheckpointReader(data_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        s = key[:len(key)-len(key.split('/')[-1])-1]
        if (key not in skip_layer) and (key.split('/')[-1] == 'weights'):
            with tf.variable_scope(s, reuse=True):
                session.run(tf.get_variable(key.split('/')[-1]).assign(reader.get_tensor(key)))
    elapsed = (time.clock() - s_time)
    elapsed = round(elapsed / 60 , 2)
    print('Loading time used:', elapsed, 'min')

def print_tensor_names(ckpt_path):
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print('tensor_name:', key)

def balanced_train_sampling(data_list,label_list):
    image = []
    label = []
    ill_list = []
    helth_list = []
    for i in range(len(label_list)):
        if label_list[i]:
            ill_list.append(i)
        else:
            helth_list.append(i)
    if len(helth_list) > len(ill_list):
        m = len(helth_list) // len(ill_list)
        r = len(helth_list) % len(ill_list)
        if m > 1:
            ill_list.extend(ill_list * (m - 1))
        idex = np.random.permutation(len(ill_list))
        for j in range(r):
            ill_list.append(ill_list[idex[j]])
    elif len(helth_list) < len(ill_list):
        m = len(ill_list) // len(helth_list)
        r = len(ill_list) % len(helth_list)
        if m > 1:
            helth_list.extend(helth_list * (m - 1))
        idex = np.random.permutation(len(helth_list))
        for j in range(r):
            helth_list.append(helth_list[idex[j]])
    for i in range(len(ill_list)):
        image.append(data_list[ill_list[i]])
        label.append(1)
        image.append(data_list[helth_list[i]])
        label.append(0)
    return image,label

def dense_crf(img, output_probs_1, output_probs_0):
    h = output_probs_1.shape[0]
    w = output_probs_1.shape[1]
    output_probs_1 = np.expand_dims(output_probs_1, 0)
    output_probs_0 = np.expand_dims(output_probs_0, 0)
    output_probs = np.append(output_probs_0, output_probs_1, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img).astype(np.uint8)
    d.setUnaryEnergy(U)
    
    gaussian_energy = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
    d.addPairwiseEnergy(gaussian_energy, compat=3)
    pairwise_energy = create_pairwise_bilateral(sdims=(50,50), schan=(5,), img=img, chdim=2)
    d.addPairwiseEnergy(pairwise_energy, compat=10)
    
    Q = d.inference(3)
    map_q = np.array(Q)[1,:].reshape(h, w)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q
