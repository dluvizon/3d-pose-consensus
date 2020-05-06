
def structural_loss_dst68j3d(p_pred, v_pred):

    v_pred = K.stop_gradient(v_pred)

    def getlength(v):
        return K.sqrt(K.sum(K.square(v), axis=-1))

    """Arms segments"""
    joints_arms = p_pred[:, :, 16:37+1, :]
    conf_arms = v_pred[:, :, 16:37+1]

    diff_arms_r = joints_arms[:, :, 2:-1:2, :] - joints_arms[:, :, 0:-3:2, :]
    diff_arms_l = joints_arms[:, :, 3::2, :] - joints_arms[:, :, 1:-2:2, :]
    c2_arms_r = conf_arms[:, :, 2:-1:2] * conf_arms[:, :, 0:-3:2]
    c2_arms_l = conf_arms[:, :, 3::2] * conf_arms[:, :, 1:-2:2]

    """Legs segments"""
    joints_legs = p_pred[:, :, 48:67+1, :]
    conf_legs = v_pred[:, :, 48:67+1]
    diff_legs_r = joints_legs[:, :, 2:-1:2, :] - joints_legs[:, :, 0:-3:2, :]
    diff_legs_l = joints_legs[:, :, 3::2, :] - joints_legs[:, :, 1:-2:2, :]
    c2_legs_r = conf_legs[:, :, 2:-1:2] * conf_legs[:, :, 0:-3:2]
    c2_legs_l = conf_legs[:, :, 3::2] * conf_legs[:, :, 1:-2:2]

    """Limbs segments"""
    segs_limbs_r = getlength(K.concatenate([diff_arms_r, diff_legs_r], axis=-2))
    segs_limbs_l = getlength(K.concatenate([diff_arms_l, diff_legs_l], axis=-2))
    c2_limbs_r = K.concatenate([c2_arms_r, c2_legs_r], axis=-1)
    c2_limbs_l = K.concatenate([c2_arms_l, c2_legs_l], axis=-1)

    len_upperarm_r = K.sum(segs_limbs_r[:, :, 2:5], axis=-1, keepdims=True)
    len_upperarm_l = K.sum(segs_limbs_l[:, :, 2:5], axis=-1, keepdims=True)
    len_forearm_r = K.sum(segs_limbs_r[:, :, 5:8], axis=-1, keepdims=True)
    len_forearm_l = K.sum(segs_limbs_l[:, :, 5:8], axis=-1, keepdims=True)
    len_hand_r = K.sum(segs_limbs_r[:, :, 8:10], axis=-1, keepdims=True)
    len_hand_l = K.sum(segs_limbs_r[:, :, 8:10], axis=-1, keepdims=True)

    c2_upperarm_r = K.sum(c2_limbs_r[:, :, 2:5], axis=-1, keepdims=True)
    c2_upperarm_l = K.sum(c2_limbs_l[:, :, 2:5], axis=-1, keepdims=True)
    c2_forearm_r = K.sum(c2_limbs_r[:, :, 5:8], axis=-1, keepdims=True)
    c2_forearm_l = K.sum(c2_limbs_l[:, :, 5:8], axis=-1, keepdims=True)
    c2_hand_r = K.sum(c2_limbs_r[:, :, 8:10], axis=-1, keepdims=True)
    c2_hand_l = K.sum(c2_limbs_r[:, :, 8:10], axis=-1, keepdims=True)

    len_femur_r = K.sum(K.concatenate([
        segs_limbs_r[:, :, 10:11],
        segs_limbs_r[:, :, 12:14],
        ], axis=-1), axis=-1, keepdims=True)
    len_femur_l = K.sum(K.concatenate([
        segs_limbs_l[:, :, 10:11],
        segs_limbs_l[:, :, 12:14],
        ], axis=-1), axis=-1, keepdims=True)

    c2_femur_r = K.sum(K.concatenate([
        c2_limbs_r[:, :, 10:11],
        c2_limbs_r[:, :, 12:14],
        ], axis=-1), axis=-1, keepdims=True)
    c2_femur_l = K.sum(K.concatenate([
        c2_limbs_l[:, :, 10:11],
        c2_limbs_l[:, :, 12:14],
        ], axis=-1), axis=-1, keepdims=True)

    len_shin_r = K.sum(segs_limbs_r[:, :, 14:17], axis=-1, keepdims=True)
    len_shin_l = K.sum(segs_limbs_l[:, :, 14:17], axis=-1, keepdims=True)
    len_feet_r = K.sum(segs_limbs_r[:, :, 17:19], axis=-1, keepdims=True)
    len_feet_l = K.sum(segs_limbs_l[:, :, 17:19], axis=-1, keepdims=True)

    c2_shin_r = K.sum(c2_limbs_r[:, :, 14:17], axis=-1, keepdims=True)
    c2_shin_l = K.sum(c2_limbs_l[:, :, 14:17], axis=-1, keepdims=True)
    c2_feet_r = K.sum(c2_limbs_r[:, :, 17:19], axis=-1, keepdims=True)
    c2_feet_l = K.sum(c2_limbs_l[:, :, 17:19], axis=-1, keepdims=True)


    joints_head = K.concatenate([
        p_pred[:, :, 11:11+1, :], p_pred[:, :, 11:11+1, :],
        p_pred[:, :, 12:15+1, :],
        p_pred[:, :, 8:8+1, :], p_pred[:, :, 8:8+1, :],
        p_pred[:, :, 14:15+1, :],
        ], axis=-2)
    conf_head = K.concatenate([
        v_pred[:, :, 11:11+1], v_pred[:, :, 11:11+1],
        v_pred[:, :, 12:15+1],
        v_pred[:, :, 8:8+1], v_pred[:, :, 8:8+1],
        v_pred[:, :, 14:15+1],
        ], axis=-1)

    diff_head_r = joints_head[:, :, 2:-1:2, :] - joints_head[:, :, 0:-3:2, :]
    diff_head_l = joints_head[:, :, 3::2, :] - joints_head[:, :, 1:-2:2, :]

    c2_head_r = conf_head[:, :, 2:-1:2] * conf_head[:, :, 0:-3:2]
    c2_head_l = conf_head[:, :, 3::2] * conf_head[:, :, 1:-2:2]

    diff_cross_r = K.concatenate([
        p_pred[:, :, 3:3+1, :] - p_pred[:, :, 20:20+1, :],
        p_pred[:, :, 49:49+1, :] - p_pred[:, :, 3:3+1, :],
        ], axis=-2)
    diff_cross_l = K.concatenate([
        p_pred[:, :, 3:3+1, :] - p_pred[:, :, 21:21+1, :],
        p_pred[:, :, 48:48+1, :] - p_pred[:, :, 3:3+1, :],
        ], axis=-2)

    diff_spine = K.concatenate([
        p_pred[:, :, 0:0+1, :] - p_pred[:, :, 7:7+1, :], # euclidean
        p_pred[:, :, 1:7+1, :] - p_pred[:, :, 0:6+1, :], # geodesic
        ], axis=-2)

    segs_spine = getlength(diff_spine)
    spine_euclidian = K.stop_gradient(segs_spine[:, :, :1])
    len_spine = K.sum(segs_spine[:, :, 1:], axis=-1, keepdims=True)

    segs_midhead = getlength(p_pred[:, :, 9:11+1, :] - p_pred[:, :, 8:10+1, :])
    len_midhead = K.sum(segs_midhead, axis=-1, keepdims=True)

    segs_ears = getlength(K.concatenate([
        p_pred[:, :, 12:12+1, :] - p_pred[:, :, 14:14+1, :],
        p_pred[:, :, 9:9+1, :] - p_pred[:, :, 12:12+1, :],
        p_pred[:, :, 13:13+1, :] - p_pred[:, :, 9:9+1, :],
        p_pred[:, :, 15:15+1, :] - p_pred[:, :, 13:13+1, :]
        ], axis=-2))
    len_ears = K.sum(segs_ears, axis=-1, keepdims=True)

    len_cross_r = K.sum(getlength(diff_cross_r), axis=-1, keepdims=True)
    len_cross_l = K.sum(getlength(diff_cross_l), axis=-1, keepdims=True)

    ref_length = K.stop_gradient(
            K.clip((len_cross_r + len_cross_l) / 2., 0.1, 1.))

    """Reference lengths based on ground truth poses from Human3.6M:
        Spine wrt. ref:         0.715   (0.032 std.)
        Spine wrt. euclidean:   1.430 (maximum) (0.046 std.)
        MidHead wrt. ref:       0.266   (0.019 std.)
        Shoulder wrt. ref:      0.150   (?? std.)
        Upper arms wrt. ref:    0.364   (0.019 std.)
        Fore arms wrt. ref:     0.326   (0.025 std.)
        Hands wrt. ref:         0.155   (0.014 std.)
        Femur wrt. ref:         0.721   (0.040 std.)
        Shin wrt. ref:          0.549   (0.063 std.)
        Feet wrt. ref:          0.294   (0.060 std.)
    """

    rules_loss = K.concatenate([
        c2_limbs_r * c2_limbs_l * (segs_limbs_r - segs_limbs_l),
        len_spine - 0.715 * ref_length,
        len_midhead - 0.266 * ref_length,
        c2_upperarm_r * (len_upperarm_r - 0.364 * ref_length),
        c2_upperarm_l * (len_upperarm_l - 0.364 * ref_length),
        c2_forearm_r * (len_forearm_r  - 0.326 * ref_length),
        c2_forearm_l * (len_forearm_l  - 0.326 * ref_length),
        c2_hand_r * (len_hand_r     - 0.155 * ref_length),
        c2_hand_l * (len_hand_l     - 0.155 * ref_length),
        c2_femur_r * (len_femur_r    - 0.721 * ref_length),
        c2_femur_l * (len_femur_l    - 0.721 * ref_length),
        c2_shin_r * (len_shin_r     - 0.549 * ref_length),
        c2_shin_l * (len_shin_l     - 0.549 * ref_length),
        c2_feet_r * (len_feet_r     - 0.294 * ref_length),
        c2_feet_l * (len_feet_l     - 0.294 * ref_length),
        len_ears       - 0.213 * ref_length,
        ], axis=-1)

    rules = K.sum(K.square(rules_loss), axis=-1)
    spine_bent = K.squeeze(K.maximum(0., len_spine - 1.430 * spine_euclidian),
            axis=-1)

    return K.mean(spine_bent + rules, axis=-1)
