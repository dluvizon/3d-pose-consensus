from keras import backend as K
import tensorflow as tf

def pose_mae_loss(y_true, y_pred):
    p_true = y_true[:, :, 0:3]
    p_pred = y_pred[:, :, 0:3]

    idx = K.cast(K.greater(p_true, -1e6), 'float32')
    l1 = K.abs(p_pred - p_true)

    loss = K.mean(idx * l1, axis=(1, 2))

    return loss


def pose_loss(y_true, y_pred, visibility_loss=True):
    """Expected y_true/y_pred shape as (batches, joints, dim+1).
    """
    p_true = y_true[:, :, 0:3]
    p_pred = y_pred[:, :, 0:3]
    if visibility_loss:
        v_true = y_true[:, :, 3]
        v_pred = y_pred[:, :, 3]

    idx = K.cast(K.greater(p_true, -1e6), 'float32')

    l1 = K.abs(p_pred - p_true)
    l2 = K.square(p_pred - p_true)

    loss = K.sum(idx * (l1 + l2), axis=(1, 2))

    if visibility_loss:
        idx = K.cast(K.greater(v_true, -1e6), 'float32')
        bc = 0.01 * K.binary_crossentropy(v_true, v_pred)
        loss += K.sum(idx * bc, axis=(1))

    return loss


def absz_loss(z_true, z_pred):
    """Expected z_true/z_pred shape as (batches, 1).
    """
    idx = K.cast(K.greater_equal(z_true, 0), 'float32')

    l1 = K.abs(z_pred - z_true)
    l2 = K.square(z_pred - z_true)

    return K.sum(idx * (l1 + l2), axis=-1)

