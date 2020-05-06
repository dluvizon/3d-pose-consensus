import numpy as np

import keras.backend as K
from keras.models import Model

from keras.applications.resnet50 import ResNet50

from keras import layers
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras.layers import Conv2D
from keras.layers import ZeroPadding2D
from keras.layers import DepthwiseConv2D
from keras.layers import SeparableConv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import UpSampling2D
from keras.layers import Reshape

from .poseregression import multipose_regression
from .poseregression import multipose_absz
from .poseregression import multiperson_prediction
from .utils import *


def relu(x, name=None):
    return Activation('relu', name=name)(x)

def conv2d(x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    return Conv2D(filters, kernel_size, strides=strides, padding=padding,
            use_bias=False, name=name)(x)


def depthwise_residual_block(inp, filters, kernel_size,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    num_filters = K.int_shape(inp)[-1]
    if isinstance(kernel_size, tuple):
        padd = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padd = kernel_size // 2

    skip = inp
    if (num_filters != filters) or (strides not in [1, (1, 1)]):
        skip = BatchNormalization(name='skip_%d_bn' % block_id)(skip)
        skip = relu(skip, name='skip_%d_relu' % block_id)
        skip = conv2d(skip, filters, 1, strides=strides,
                name='skip_%d' % block_id)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(inp)
    x = relu(x, name='conv_dw_%d_relu' % block_id)
    x = ZeroPadding2D(padding=padd, name='conv_pad_%d' % block_id)(x)
    x = DepthwiseConv2D(kernel_size,
                        padding='valid',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(x)

    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = relu(x, name='conv_pw_%d_relu' % block_id)
    x = conv2d(x, filters, 1, name='conv_pw_%d' % block_id)

    return layers.add([x, skip])



def densely_connected_unet(lx, growth, block_id=1):
    if 'dwres_cnt' not in globals():
        global dwres_cnt
        dwres_cnt = 0

    """Downscaling part."""
    x = lx[0]
    for i in range(1, len(lx)):
        num_features = K.int_shape(x)[-1] + growth

        x = MaxPooling2D(2, 2, padding='same',
                name='unet_%d_maxpooling_%d' % (block_id, i))(x)

        ks = 5 if i < len(lx) - 1 else 3

        dwres_cnt += 1
        x = depthwise_residual_block(x, num_features, ks, block_id=dwres_cnt)
        if lx[i] is not None:
            x = layers.add([x, lx[i]])

        dwres_cnt += 1
        x = depthwise_residual_block(x, num_features, ks, block_id=dwres_cnt)

        lx[i] = x

    """Upscaling part."""
    x = lx[-1]
    for i in range(len(lx)-1)[::-1]:
        num_features = K.int_shape(x)[-1] - growth

        x = UpSampling2D(2, name='unet_%d_upsampling_%d' % (block_id, i))(x)

        ks = 5

        dwres_cnt += 1
        x = depthwise_residual_block(x, num_features, ks, block_id=dwres_cnt)
        if lx[i] is not None:
            x = layers.add([x, lx[i]])

        dwres_cnt += 1
        x = depthwise_residual_block(x, num_features, ks, block_id=dwres_cnt)

        lx[i] = x

    return x


def resnet50_backbone(x, last_layer=142):
    """ResNet50 backbone. Last layer can be 80, 142 or 174.
    """

    resnet = ResNet50(include_top=False, input_tensor=x, weights='imagenet')
    outputs = []

    printnl('ResNet50 backbone:')
    x = resnet.layers[last_layer].output
    printnl('  layer_%d: ' % last_layer + x.name + ':\t' + str(x))

    return x


def head_network(x, num_ups, growth):
    """Head network for pose estimation.

    # Arguments
        inputs: Input tensor.
        num_ups: Number of upscalings
        growth: Growth rate (incremental)

    # Return
        Output tensor and visual features tensor (for abs z).
    """

    assert num_ups in [0, 1, 2], 'Invalid num_ups {}'.format(num_ups)

    if num_ups == 0:
        vfeat = x
        x = conv2d(x, 4*growth, (1, 1), name='headnet_%d_conv' % num_ups)

        return x, vfeat

    x = Conv2DTranspose(3*growth, (2, 2), strides=(2, 2), padding='same',
            use_bias=False, name='headnet_1_convT')(x)
    x = BatchNormalization(name='headnet_1_bn')(x)
    x = Activation('relu', name='headnet_1_relu')(x)
    vfeat = x

    x = DepthwiseConv2D(5, padding='same', use_bias=False,
            name='headnet_1_dw_conv')(x)

    if num_ups == 2:
        x = BatchNormalization(name='headnet_1_dw_bn')(x)
        x = Activation('relu', name='headnet_1_dw_relu')(x)

        x = Conv2DTranspose(2*growth, (2, 2), strides=(2, 2), padding='same',
                use_bias=False, name='headnet_2_convT')(x)
        x = BatchNormalization(name='headnet_2_bn')(x)
        x = Activation('relu', name='headnet_2_relu')(x)
        vfeat = x

        x = DepthwiseConv2D(5, padding='same', use_bias=False,
                name='headnet_2_dw_conv')(x)

    return x, vfeat


def residual(x, num_filters=None, name=None):
    """Implements a Residual Unit with standard convolutions. """

    if num_filters is None:
        num_filters = K.int_shape(x)[-1]

    if num_filters == K.int_shape(x)[-1]:
        shortcut = x

    x = BatchNormalization(name=appstr(name, '_bn1'))(x)
    x = Activation('relu', name=appstr(name, '_act1'))(x)

    if num_filters != K.int_shape(x)[-1]:
        shortcut = Conv2D(num_filters, (1, 1), use_bias=False,
                name=appstr(name, '_shortcut'))(x)

    x = Conv2D(int(num_filters / 2), (1, 1), use_bias=False,
            name=appstr(name, '_conv1'))(x)

    x = BatchNormalization(name=appstr(name, '_bn2'))(x)
    x = Activation('relu', name=appstr(name, '_act2'))(x)
    x = Conv2D(int(num_filters / 2), (3, 3), use_bias=False, padding='same',
            name=appstr(name, '_conv2'))(x)

    x = BatchNormalization(name=appstr(name, '_bn3'))(x)
    x = Activation('relu', name=appstr(name, '_act3'))(x)
    x = Conv2D(num_filters, (1, 1), use_bias=False,
            name=appstr(name, '_conv3'))(x)

    x = layers.add([x, shortcut])

    return x


def shg_entry_flow(x, image_div, num_filters):

    assert (image_div & (image_div - 1) == 0) and image_div >= 4, \
            'Invalid image_div ({}).'.format(image_div)

    x = Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, name='conv1')(x)
    x = residual(x, 128, name='res1')
    x = MaxPooling2D(3, 2, padding='same', name='maxpooling1')(x)

    x = residual(x, 128, name='res2')
    x = residual(x, num_filters, name='res3')

    res_cnt = 3
    div_factor = 4

    while div_factor < image_div:
        x = MaxPooling2D(2, 2, padding='same',
                name='maxpooling%d' % int(res_cnt//2 + 1))(x)

        x = residual(x, name='res%d' % (res_cnt + 1))
        x = residual(x, name='res%d' % (res_cnt + 2))

        res_cnt += 2
        div_factor *= 2

    return x


def hourglass(x, num_levels, name=None):

    up = residual(x, name=appstr(name, '_rup%d' % num_levels))

    low = MaxPooling2D(2, 2, padding='same',
            name=appstr(name, '_maxpooling%d' % num_levels))(x)

    low = residual(low, name=appstr(name, '_rlow1.%d' % num_levels))
    if num_levels > 1:
        low = hourglass(low, num_levels-1, name=name)
    else:
        low = residual(low, name=appstr(name, '_rlow2.%d' % num_levels))
    low = residual(low, name=appstr(name, '_rlow3.%d' % num_levels))
    low = UpSampling2D(2, name=appstr(name, '_upsampling%d' % num_levels))(low)

    x = layers.add([up, low])

    return x


def People3D(input_shape, anchors, num_joints,
        image_div=8,
        growth=128,
        num_levels=4,
        num_predictions=2,
        max_overlapping=1,
        basemodel='resnet50',
        legacy_model=True,
        dbg_heatmaps=False,
        output_vfeat=False):

    input_res = np.array(input_shape[0:2])
    min_size = image_div * max(anchors[0])
    step = image_div * (2**(num_levels - 1))

    assert (input_res >= min_size).all() and (input_res % step == 0).all(), \
            'Invalid input_shape {}'.format(input_shape)

    inp_frame = Input(shape=input_shape)
    outputs = []
    dbg_hlist = []

    if max_overlapping > 1:
        anchors = max_overlapping * anchors

    if basemodel == 'resnet50':
        x = resnet50_backbone(inp_frame)
        num_ups = int(np.log2(256 // image_div) - 4)
        x, vfeat = head_network(x, num_ups=num_ups, growth=growth)

        inp_aref = Input(shape=(4,))

        lx = [None for _ in range(num_levels)]
        lx[0] = x

        for bidx in range(num_predictions):
            if bidx > 0:
                x = densely_connected_unet(lx, growth, block_id=2*bidx+1)
                x = densely_connected_unet(lx, growth, block_id=2*bidx+2)
            ident = x

            poses, x, hlist = multiperson_prediction(x, num_joints,
                    block_id=(bidx+1))
            if legacy_model:
                ident = layers.add([ident, x])
            else:
                lx[0] = layers.add([ident, x, lx[0]])

            outputs.append(poses)
            dbg_hlist += hlist

    elif basemodel == 'shg':
        x = shg_entry_flow(inp_frame, image_div, 3*growth)
        ident = x
        vfeat = x

        inp_aref = Input(shape=(4,))

        for bidx in range(num_predictions):
            x = hourglass(ident, num_levels, name='hg%d' % (bidx+1))

            poses, x, hlist = multiperson_prediction(x, num_joints,
                    block_id=(bidx+1))
            ident = layers.add([ident, x])

            outputs.append(poses)
            dbg_hlist += hlist

    else:
        raise ValueError('Invalid `basemodel` {}'.format(basemodel))

    if output_vfeat:
        outputs.append(vfeat)

    """Predict the absolute Z based on aref and zf, for each anchor."""
    absz = multipose_absz(inp_aref, vfeat, anchors)

    modelname = 'People3D_' + basemodel
    if dbg_heatmaps:
        model = Model([inp_frame, inp_aref], absz + outputs + dbg_hlist,
                name=modelname + '_dbg')
    else:
        model = Model([inp_frame, inp_aref], absz + outputs, name=modelname)

    return model


