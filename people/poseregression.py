import numpy as np
from scipy.ndimage import gaussian_filter

import keras.backend as K

from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import SeparableConv2D
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import TimeDistributed
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D

from .activations import exponential
from .activations import channel_softmax_2d

from .utils import printnl


def appstr(s, a):
    """Safe appending strings."""
    try:
        return s + a
    except:
        return None

def linspace_2d(nb_rols, nb_cols, axis=0, vmin=0., vmax=1.):

    def _lin_sp_aux(size, nb_repeat):
        linsp = np.linspace(vmin, vmax, num=size)
        x = np.empty((nb_repeat, size), dtype=np.float32)

        for d in range(nb_repeat):
            x[d] = linsp

        return x

    if axis == 1:
        return (_lin_sp_aux(nb_rols, nb_cols)).T
    return _lin_sp_aux(nb_cols, nb_rols)


def kernel_expectation_2d(x, kernel_size, axis, vmin=0., vmax=1., name=None):

    """Implements a 2D linear interpolation (x for axis=0 and y for axis=1)
    using a depthwise convolution (non trainable).

    # Arguments
        x: Input tensor (None, H, W, num_points)
        kernel_size: tuple (h, w)

    # Return
        Tensor (None, H-h+1, W-w+1, num_points)
    """
    assert K.ndim(x) == 4, 'Input tensor must have ndim 4 {}'.format(K.ndim(x))

    if 'global_sam_cnt' not in globals():
        global global_sam_cnt
        global_sam_cnt = 0

    if name is None:
        name = '_%d' % global_sam_cnt
        global_sam_cnt += 1
    name = name + '%dx%d' % kernel_size

    num_filters = K.int_shape(x)[-1]

    lins = np.expand_dims(linspace_2d(kernel_size[0], kernel_size[1],
        axis=axis, vmin=vmin, vmax=vmax), axis=-1)

    if num_filters > 1:
        lins = np.tile(lins, (1, 1, num_filters))

    f = DepthwiseConv2D(kernel_size,
                        padding='valid',
                        depth_multiplier=1,
                        strides=1,
                        use_bias=False,
                        name=name)
    x = f(x)

    wx = f.get_weights()
    wx[0][:, :, :, 0] = lins
    f.set_weights(wx)
    f.trainable = False

    return x


def normalize_features(x, norm, name=None):
    def _norm(inputs):
        return inputs[0] / K.clip(inputs[1], 1e-4, None)
    return Lambda(_norm, name=name)([x, norm])


def decompose_identity(x, num_anchors, num_joints):
    def _dec_ident(x):
        x = K.reshape(x, (-1, num_anchors * num_anchors, num_joints))
        return x[:, 0::num_anchors+1, :]

    return Lambda(_dec_ident)(x)


def kernel_sum(x, kernel_size, strides=1, padding='valid', name=None):

    f = DepthwiseConv2D(kernel_size,
            padding=padding,
            depth_multiplier=1,
            strides=strides,
            use_bias=False,
            name=name)

    x = f(x)
    w = f.get_weights()
    w[0][:] = 1.
    f.set_weights(w)
    f.trainable = False

    return x


def anchored_softargmax_2d(h, anchor, name_postfix,
        vmin=(0., 0.), vmax=(1., 1.)):

    eh = Activation(exponential, name='heatmap_exp_' + name_postfix)(h)
    seh = kernel_sum(eh, anchor, name='fixed_conv_sum_eh_' + name_postfix)

    xx = kernel_expectation_2d(eh, anchor, axis=0, vmin=vmin[1], vmax=vmax[1],
            name='fixed_sam_x_' + name_postfix)
    yy = kernel_expectation_2d(eh, anchor, axis=1, vmin=vmin[0], vmax=vmax[0],
            name='fixed_sam_y_' + name_postfix)

    xx = normalize_features(xx, seh)
    yy = normalize_features(yy, seh)

    return xx, yy, eh, seh


def anchored_softargmax_3d(h, name_postfix):
    pass


def keypoint_confidence_2d(h, eh, seh, anchor, name_postfix, sigma=0.5):

    s = MaxPooling2D(anchor, strides=(1, 1), padding='valid',
            name='h_maxpooling_' + name_postfix)(h)
    s = Activation('sigmoid', name='heatmaps_sig_' + name_postfix)(s)

    cc = MaxPooling2D(anchor, strides=(1, 1), padding='valid',
            name='conf_maxpooling_' + name_postfix)(eh)

    cc = normalize_features(cc, seh)

    from .utils.pose import CONF_MIN_VAL, CONF_MAX_VAL
    cc = Lambda(lambda x: K.clip(x, CONF_MIN_VAL, CONF_MAX_VAL))(cc)

    cc = layers.multiply([cc, s])

    return cc


def lin_interpolation_2d(x, axis, vmin=0., vmax=1., name=None):
    """Implements a 2D linear interpolation using a depth size separable
    convolution (non trainable).
    """
    assert K.ndim(x) in [4, 5], \
            'Input tensor must have ndim 4 or 5 ({})'.format(K.ndim(x))

    if 'global_sam_cnt' not in globals():
        global global_sam_cnt
        global_sam_cnt = 0

    if name is None:
        name = 'custom_sam_%d' % global_sam_cnt
        global_sam_cnt += 1

    if K.ndim(x) == 4:
        num_rows, num_cols, num_filters = K.int_shape(x)[1:]
    else:
        num_rows, num_cols, num_filters = K.int_shape(x)[2:]

    f = SeparableConv2D(num_filters, (num_rows, num_cols), use_bias=False,
            name=name)
    x = TimeDistributed(f, name=name)(x) if K.ndim(x) == 5 else f(x)

    w = f.get_weights()
    w[0].fill(0)
    w[1].fill(0)
    linspace = linspace_2d(num_rows, num_cols, axis=axis)

    for i in range(num_filters):
        w[0][:,:, i, 0] = linspace[:,:]
        w[1][0, 0, i, i] = 1.

    f.set_weights(w)
    f.trainable = False

    x = Lambda(lambda x: K.squeeze(x, axis=-2))(x)
    x = Lambda(lambda x: K.squeeze(x, axis=-2))(x)
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)

    return x


def softargmax2d(x, limits=(0, 0, 1, 1), name=None):
    x_x = lin_interpolation_2d(x, axis=0, vmin=limits[0], vmax=limits[2],
            name=appstr(name, '_x'))
    x_y = lin_interpolation_2d(x, axis=1, vmin=limits[1], vmax=limits[3],
            name=appstr(name, '_y'))
    x = layers.concatenate([x_x, x_y], name=name)

    return x


def keypoint_confidence(x, name=None):
    """Implements the keypoint (body joint) confidence, given a set of
    probability maps as input. No parameters required.
    """
    def _keypoint_confidence(x):
        x = 4 * AveragePooling2D((2, 2), strides=(1, 1))(x)
        x = K.expand_dims(GlobalMaxPooling2D()(x), axis=-1)

        return x

    f = Lambda(_keypoint_confidence, name=name)

    return TimeDistributed(f, name=name)(x) if K.ndim(x) == 5 else f(x)


def multipose_regression(h, d, block_id=1):
    """Implement a multipose regression.

    # Arguments
        h: Heatmaps tensor (None, H, W, num_joints).
        d: Depth maps tensor (None, H, W, num_joints), only used in
        conjuntion with 2D soft-argmax.

    # Return
        Tensor with pose predictions (None, num_anchors, num_joints, dim) for
            2D soft-argmax, or (None, num_joints, dim) for 3D soft-argmax.
    """

    assert K.ndim(h) == 4, 'Invalid heatmap shape {}'.format(K.int_shape(h))

    vmin = 0
    vmax = 1

    hs = Activation(channel_softmax_2d())(h)
    p = softargmax2d(hs)
    c = keypoint_confidence(hs)

    d = Lambda(lambda x: K.sigmoid(0.5*x))(d)
    z = layers.multiply([d, hs])

    z = Lambda(lambda x: K.sum(x, axis=(-2, -3)))(z)
    z = Lambda(lambda x: K.expand_dims(x, axis=-1))(z)
    p = layers.concatenate([p, z, c], name='p%d' % block_id)

    return p


def multipose_absz(inp_aref, vfeat, anchors):
    """Expected inp_aref shape as (None, num_anchors, 4).
    """

    aref =  Dense(128, input_shape=K.int_shape(inp_aref)[1:],
            name='td_fc_aref')(inp_aref)

    vfeat = Conv2D(128, 1, use_bias=False, name='branch_vfeat_conv')(vfeat)
    vfeat = GlobalAveragePooling2D()(vfeat)

    x = layers.concatenate([vfeat, aref], axis=-1, name='abs_z_concat')
    ident = x

    x = BatchNormalization(name='abs_z1_bn')(x)
    x = LeakyReLU(0.1, name='abs_z1_act')(x)
    x = Dense(256, name='abs_z1_tdfc')(x)

    x = BatchNormalization(name='abs_z2_bn')(x)
    x = LeakyReLU(0.1, name='abs_z2_act')(x)
    x = Dense(256, name='abs_z2_tdfc')(x)

    x = layers.add([ident, x])
    x = BatchNormalization(name='abs_z3_bn')(x)
    x = LeakyReLU(0.1, name='abs_z3_act')(x)
    x = Dense(1, name='abs_z3_tdfc')(x)
    # x = Dense(1, name='za2')(x)

    x = Lambda(lambda x: K.sigmoid(0.5*x), name='sza')(x)

    return [x]


def multiperson_prediction(x, num_joints, block_id=1):

    num_features = K.int_shape(x)[-1]
    reinject = []
    poses = []
    hlist = []

    x = BatchNormalization(name='pred_dw_%d_bn' % block_id)(x)
    x = Activation('relu', name='pred_dw_%d_relu' % block_id)(x)
    x = ZeroPadding2D(padding=(1, 1), name='pred_pad_%d' % block_id)(x)

    x1 = SeparableConv2D(num_features, (3, 3),
                        padding='valid',
                        strides=1,
                        use_bias=False,
                        name='pred_dw1_%d' % block_id)(x)
    reinject.append(x1)
    x1 = BatchNormalization(name='pred_pw1_%d_bn' % block_id)(x1)
    x1 = Activation('relu', name='pred_pw1_%d_relu' % block_id)(x1)

    x2 = SeparableConv2D(num_features, (3, 3),
                        padding='valid',
                        strides=1,
                        use_bias=False,
                        name='pred_dw2_%d' % block_id)(x)
    reinject.append(x2)
    x2 = BatchNormalization(name='pred_pw2_%d_bn' % block_id)(x2)
    x2 = Activation('relu', name='pred_pw2_%d_relu' % block_id)(x2)

    """Heatmaps prediction."""
    h = Conv2D(num_joints, 3, use_bias=True, padding='same',
            name='pred_h_%d' % (block_id))(x1)
    hlist.append(h)

    hact = Activation('relu',
            name='reinject_h_%d_relu' % (block_id))(h)
    # reinject.append(hact)

    """Depthmaps prediction."""
    d = Conv2D(num_joints, 3, use_bias=True, padding='same',
            name='pred_d_%d' % (block_id))(x2)
    # reinject.append(d)

    # d = Activation('sigmoid')(d)

    p = multipose_regression(h, d, block_id=block_id)
    poses.append(p)

    # x = layers.concatenate(reinject, name='reinject_concat_%d' % block_id)
    x = layers.add(reinject, name='reinject_sum_%d' % block_id)
    x = Conv2D(num_features, 3, use_bias=False, padding='same',
                name='reinject_%d' % block_id)(x)

    poses = poses[0]

    return poses, x, hlist

