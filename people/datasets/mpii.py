import os

import numpy as np
import scipy.io as sio
from PIL import Image

from .generic import GenericDataset
from .generic import project_gt_poses_to_anchors
from .generic import compute_anchors_reference
from .generic import compute_window_reference
from ..utils import *


def load_mpii_mat_annotation(filename):
    mat = sio.loadmat(filename)
    annot_tr = mat['annot_tr']
    annot_val = mat['annot_val']

    # Respect the order of TEST (0), TRAIN (1), and VALID (2)
    rectidxs = [None, annot_tr[0,:], annot_val[0,:]]
    images = [None, # e.g. 038279059.jpg
            [int(s[0][:9]) for s in annot_tr[1,:]],
            [int(s[0][:9]) for s in annot_val[1,:]]]
    annorect = [None, annot_tr[2,:], annot_val[2,:]]

    return rectidxs, images, annorect


def serialize_annorect(rectidxs, annorect):
    assert len(rectidxs) == len(annorect)

    sample_list = []
    for i in range(len(rectidxs)):
        rec = rectidxs[i]
        for j in range(rec.size):
            idx = rec[j,0]-1 # Convert idx from Matlab
            ann = annorect[i][idx,0]
            annot = {}
            annot['head'] = ann['head'][0,0][0]
            annot['objpos'] = ann['objpos'][0,0][0]
            annot['scale'] = ann['scale'][0,0][0,0]
            annot['pose'] = ann['pose'][0,0]
            annot['imgidx'] = i
            sample_list.append(annot)

    return sample_list


def match_images_and_poses(sample_list, image_list):

    """Create a dictionary to map all poses for a given image."""
    img2pose = {}
    for i, annot in enumerate(sample_list):
        imgidx = annot['imgidx']
        p = np.expand_dims(annot['pose'].T, axis=0)
        obj = np.expand_dims(annot['objpos'], axis=0)

        if imgidx not in img2pose:
            img2pose[imgidx] = []
        img2pose[imgidx].append((p, obj))

    """Copy the pose list for each image to the concerned samples."""
    for annot in sample_list:
        imgidx = annot['imgidx']
        data = img2pose[imgidx]
        if len(data) > 1:
            p = np.concatenate([x[0] for x in data], axis=0)
            obj = np.concatenate([x[1] for x in data], axis=0)
        else:
            p = data[0][0]
            obj = data[0][1]

        annot['pose'] = p
        annot['objpos'] = obj


def calc_head_size(head_annot):
    head = np.array([float(head_annot[0]), float(head_annot[1]),
        float(head_annot[2]), float(head_annot[3])])
    return 0.6 * np.linalg.norm(head[0:2] - head[2:4])


class MPII(GenericDataset):
    """Implementation of the MPII dataset for multi-person pose estimation.
    """

    def __init__(self, dataset_path, dataconf, poselayout=pa16j2d,
            remove_outer_joints=True, preprocess_mode='tf'):

        GenericDataset.__init__(self, dataset_path, dataconf,
                poselayout=poselayout,
                remove_outer_joints=remove_outer_joints,
                preprocess_mode=preprocess_mode)

        self.load_annotations(os.path.join(dataset_path, 'annotations.mat'))


    def load_annotations(self, filename):
        try:
            rectidxs, images, annorect = load_mpii_mat_annotation(filename)

            self.samples = {}
            self.samples[TEST_MODE] = [] # No samples for test
            self.samples[TRAIN_MODE] = serialize_annorect(
                    rectidxs[TRAIN_MODE], annorect[TRAIN_MODE])
            self.samples[VALID_MODE] = serialize_annorect(
                    rectidxs[VALID_MODE], annorect[VALID_MODE])
            self.images = images

        except:
            warning('Error loading the MPII dataset!')
            raise

    def load_image(self, key, mode):
        try:
            imgidx = self.samples[mode][key]['imgidx']
            image = '%09d.jpg' % self.images[mode][imgidx]
            imgt = T(Image.open(os.path.join(
                self.dataset_path, 'images', image)))
        except:
            warning('Error loading sample key/mode: %d/%d' % (key, mode))
            raise

        return imgt

    def get_data(self, key, mode, fast_crop=False):
        pl = self.poselayout # alias for poselayout
        output = {}

        if mode == TRAIN_MODE:
            dconf = self.dataconf.random_data_generator()
        else:
            dconf = self.dataconf.get_fixed_config()

        imgt = self.load_image(key, mode)
        imgsize = imgt.size
        annot = self.samples[mode][key]

        scale = 1.25*annot['scale']

        objpos = np.array([annot['objpos'][0], annot['objpos'][1] + 12*scale])
        objpos += scale * np.array([dconf['transx'], dconf['transy']])
        winsize = 200 * dconf['scale'] * scale
        winsize = (winsize, winsize)
        output['bbox'] = objposwin_to_bbox(objpos, winsize)

        if fast_crop:
            """Slightly faster method, but gives lower precision."""
            imgt.crop_resize_rotate(objpos, winsize,
                    self.dataconf.crop_resolution, dconf['angle'])
        else:
            imgt.rotate_crop(dconf['angle'], objpos, winsize)
            imgt.resize(self.dataconf.crop_resolution)

        if dconf['hflip'] == 1:
            imgt.horizontal_flip()

        imgt.normalize_affinemap()
        frame = normalize_channels(imgt.asarray(),
                channel_power=dconf['chpower'], mode=self.preprocess_mode)

        if dconf['geoocclusion'] is not None:
            geo = dconf['geoocclusion']
            frame[geo[0]:geo[2], geo[1]:geo[3], :] = 0.
        output['frame'] = frame

        pose = np.empty((pl.num_joints, pl.dim))
        pose[:] = np.nan

        head = annot['head']
        pose[pl.map_to_mpii, 0:2] = \
                transform_2d_points(imgt.afmat, annot['pose'].T, transpose=True)
        if imgt.hflip:
            pose = pose[pl.map_hflip, :]

        # Set invalid joints and NaN values as an invalid value
        pose[np.isnan(pose)] = -1e9
        v = np.expand_dims(get_visible_joints(pose[:, 0:2]), axis=-1)
        if self.remove_outer_joints:
            pose[(v==0)[:, 0], :] = -1e9
        pose = np.concatenate([pose, v], axis=-1)

        """For 3D poses, set Z of root joints as the bbox center."""
        if pose.shape[-1] == 3:
            pose[0, 2] = 0.5

        afmat = imgt.afmat.copy()
        output['afmat'] = afmat
        output['aref'] = compute_window_reference(afmat, imgsize)
        output['pose'] = pose
        output['pose_c'] = -1e9 * np.ones(pose.shape)
        output['headsize'] = calc_head_size(annot['head'].copy())
        output['rootz'] = -1e9

        return output


    def get_shape(self, dictkey):
        if dictkey == 'frame':
            return self.dataconf.input_shape
        if dictkey == 'pose':
            return (self.poselayout.num_joints, self.poselayout.dim+1)
        if dictkey == 'rootz':
            return (1,)
        if dictkey == 'aref':
            return (4,)
        if dictkey == 'headsize':
            return (1,)
        if dictkey == 'afmat':
            return (3, 3)
        raise Exception('Invalid dictkey `{}` on get_shape!'.format(dictkey))


    def get_length(self, mode):
        return len(self.samples[mode])

