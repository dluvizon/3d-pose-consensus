import os
import sys
import copy

import json
import numpy as np
from PIL import Image

# from deephar.data import Dataset
# from deephar.utils import *
from .generic import GenericDataset
from .generic import project_gt_poses_to_anchors
from .generic import compute_anchors_reference
from ..utils import *


def coco_parser(dataset_path, data_type):

    try:
        ann_file = os.path.join(dataset_path,
                'annotations/person_keypoints_{}.json'.format(data_type))

        with open(ann_file) as json_data:
            data = json.load(json_data)
    except:
        warning('Error loading COCO file {}'.format(ann_file))
        raise

    joint_labels = data['categories'][0]['keypoints']
    images = data['images']
    annotations = []

    for ann in data['annotations']:
        if (ann['category_id'] == 1) and (np.sum(ann['keypoints']) > 0):
            annotations.append(ann)

    return images, annotations, joint_labels


def convert_bbox_from_coco(bbox):
    """Convert a bounding box array from the COCO format [x1, y1, w, h] to the
    format [x1, y1, x2, y2].
    """
    x1, y1 = (bbox[0], bbox[1])
    x2, y2 = (x1 + bbox[2], y1 + bbox[3])

    return np.array([x1, y1, x2, y2])

def compute_best_box(annot):
    xc1, yc1, xc2, yc2 = convert_bbox_from_coco(annot['bbox'])

    aux = np.reshape(np.array(annot['keypoints'], dtype=np.float), (-1, 3))
    xp1, yp1, xp2, yp2 = get_valid_bbox(aux[:, :2], aux[:, 2], relsize=1.2,
            square=False)

    """Use the maximum between the two."""
    return np.array(
            [min(xc1, xp1), min(yc1, yp1), max(xc2, xp2), max(yc2, yp2)])


def match_images_to_annotations(images, annotations):

    image_ids = np.array([img['id'] for img in images])
    annot_image_id = np.array([ann['image_id'] for ann in annotations])
    image2annot = [np.where(annot_image_id == idx) for idx in image_ids]
    annot2image = [np.where(image_ids == idx) for idx in annot_image_id]

    return image2annot, annot2image


class Coco(GenericDataset):
    """Implementation of the COCO dataset for training as data augmentation.
    """

    def __init__(self, dataset_path, dataconf, anchors_size,
            image_div=16,
            poselayout=pa16j2d,
            data_type='train2017',
            max_overlapping=1,
            topology='people',
            remove_outer_joints=True,
            preprocess_mode='tf'):

        GenericDataset.__init__(self,
                dataset_path,
                dataconf,
                anchors_size,
                image_div,
                poselayout,
                max_overlapping,
                remove_outer_joints,
                preprocess_mode)

        assert topology in ['frames', 'people'], \
                'Invalid topology {}'.format(topology)
        if topology == 'frames':
            raise ValueError('topology `frames` not implemented!')

        self.data_type = data_type
        self.topology = topology
        self._load_annotations(dataset_path, data_type)

    def _load_annotations(self, dataset_path, data_type):
        self.images, self.annot, self.joint_labels = \
                coco_parser(dataset_path, data_type)

        self.image2annot, self.annot2image = \
                match_images_to_annotations(self.images, self.annot)


    def load_image(self, image):
        try:
            imgt = T(Image.open(os.path.join(
                self.dataset_path, self.data_type, image['file_name'])))
        except:
            warning('Error loading COCO image: {}'.format(image))
            raise

        return imgt


    def get_data(self, key, mode=TRAIN_MODE):
        pl = self.poselayout # alias for poselayout
        output = {}

        if mode == TRAIN_MODE:
            dconf = self.dataconf.random_data_generator()
        else:
            dconf = self.dataconf.get_fixed_config()

        image_idx = self.annot2image[key][0][0]
        annot_idx = self.image2annot[image_idx][0]
        annotall = [self.annot[i] for i in annot_idx]

        imgt = self.load_image(self.images[image_idx])
        imsize = imgt.size

        """Using the first person to center the bbox."""
        bbox = compute_best_box(self.annot[key])
        objpos, winsize = bbox_to_objposwin(bbox)

        ratio = self.dataconf.crop_resolution[0] \
                / self.dataconf.crop_resolution[1]

        scale = 1.5 * dconf['scale']
        min_w = scale * winsize[0]
        min_h = scale * winsize[1]
        if min_w > min_h * ratio:
            min_h = min_w / ratio
        else:
            min_w = min_h * ratio

        yshift = max(self.dataconf.crop_resolution) // 21
        objpos[1] += yshift * scale
        winsize = (min_w, min_h)
        output['bbox'] = objposwin_to_bbox(objpos, winsize)

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

        def bbox2obj(b):
            return [b[0] + b[2] / 2, b[1] + b[3] / 2]

        allobj = np.array([bbox2obj(x['bbox']) for x in annotall])
        allobj = allobj.astype(float)
        allposes = np.array(
                [np.reshape(x['keypoints'], (-1, 3)) for x in annotall]).copy()
        allposes = allposes.astype(float)

        vis = allposes[:, :, 2]
        allposes = allposes[:, :, :2]
        allposes[allposes == 0] = np.nan

        """Select the poses that are closer to the anchor's center."""
        tobj = np.ones((allobj.shape[0], allobj.shape[1]+1))
        tobj[:, :-1] = transform_2d_points(imgt.afmat, allobj, transpose=True)
        tobj = project_gt_poses_to_anchors(tobj, self.base_anchors)
        dist_to_center = np.sum(np.square(tobj[:, :, :2] - 0.5), axis=-1)

        arg_sorted_dist = np.argsort(dist_to_center, axis=1)

        num_joints, dim = allposes.shape[1:]
        max_poses = self.num_anchors
        num_poses = min(max_poses,
                arg_sorted_dist.shape[1] * len(self.base_anchors))
        arg_sorted_dist = (arg_sorted_dist[:,:self.max_overlapping].T).flatten()

        tpose = np.nan * np.ones((max_poses, num_joints, dim))
        tpose[:num_poses] = allposes[arg_sorted_dist]
        tpose = np.reshape(tpose, (max_poses * num_joints, -1))
        tpose = transform_2d_points(imgt.afmat, tpose, transpose=True)
        vis = get_visible_joints(tpose[:, :2])

        tpose = np.reshape(tpose, (max_poses, num_joints, -1))
        vis = np.reshape(vis, (max_poses, num_joints))

        poses = np.nan * np.ones((max_poses, pl.num_joints, pl.dim))
        c = np.nan * np.ones((max_poses, pl.num_joints, 1))

        poses[:, pl.map_to_coco, :2] = tpose
        c[:, pl.map_to_coco, 0] = vis

        """If neighbors are defined, attribute confidence scores no nans."""
        if hasattr(pl, 'neighbors'):
            num_iter = 3 if pl == dst68j3d else 2
            c = assign_nn_confidence(c, pl.neighbors, num_iter=num_iter)

        """For 3D poses, set Z of root joints as the bbox center."""
        if poses.shape[-1] == 3:
            poses[:, 0, 2] = 0.5

        poses = np.concatenate((poses, c), axis=-1)
        if pl == dst68j3d:
            for i in range(len(poses)):
                poses[i] = dst68j3d.compute_soft_joints(poses[i])

        """Concatenate pose and confidence scores and flip it if so."""
        if imgt.hflip:
            poses = poses[:, pl.map_hflip, :]

        poses = project_gt_poses_to_anchors(poses, self.anchors)

        if self.remove_outer_joints:
            poses[(poses[:, :, -1] < 0.25), :-1] = np.nan

        """Set NaN values as invalid (-1e9)."""
        poses[np.isnan(poses)] = -1e9

        afmat = imgt.afmat.copy()
        output['afmat'] = afmat
        output['aref'] = compute_anchors_reference(self.anchors, afmat, imsize)
        output['pose'] = poses
        output['rootz'] = -1e9

        return output

    def get_shape(self, dictkey):
        if dictkey == 'frame':
            return self.dataconf.input_shape
        if dictkey == 'pose':
            return (self.num_anchors, self.poselayout.num_joints,
                    self.poselayout.dim+1)
        if dictkey == 'rootz':
            return (self.num_anchors,)
        if dictkey == 'aref':
            return (self.num_anchors, 4)
        if dictkey == 'afmat':
            return (3, 3)
        raise Exception('Invalid dictkey `{}` on get_shape!'.format(dictkey))

    def get_length(self, mode=None):
        if self.topology == 'frames':
            return len(self.images)
        elif self.topology == 'people':
            return len(self.annot)
        return 0


def convert_keypoints_from_coco(keypoints):
    """Convert an array of keypoints from the COCO format [x1, y1, v1, ...]
    to a pose [[x1, y1], [x2, y2], ...] and visible [v1, v2, ...] arrays.
    """
    aux = np.reshape(np.array(keypoints, dtype=float), (-1, 3))

    return aux[:,0:2], aux[:,2]

