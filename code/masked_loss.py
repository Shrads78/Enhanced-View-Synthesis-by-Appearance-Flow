import tensorflow as tf

def maskedl1loss(y_true, y_pred):
        gt_mask = tf.cast(y_true[:,:,:,3], tf.bool)
        masked_pred = tf.boolean_mask(y_pred, gt_mask)
        masked_gt = tf.boolean_mask(y_true[:,:,:,:-1], gt_mask)
        loss = tf.contrib.losses.absolute_difference(masked_pred,masked_gt)
        return loss
