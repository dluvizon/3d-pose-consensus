import numpy as np

from .pose import get_valid_joints

from .io import WARNING
from .io import printcn
from .io import warning

from .data_utils import TEST_MODE
from .data_utils import TRAIN_MODE
from .data_utils import VALID_MODE

relsize_std = 1.5
square_std = True

class PoseBBox():
    def __init__(self, poses, relsize=relsize_std, square=square_std):
        self.poses = poses
        self.relsize = relsize
        self.square = square
        if len(poses.shape) == 4:
            self.num_frames = poses.shape[1]
        else:
            self.num_frames = None

    def __getitem__(self, key):
        p = self.poses[key]
        if isinstance(key, int):
            return self._get_bbox(p)
        if isinstance(key, slice):
            indices = key.indices(len(self))
            key = range(*indices)
        x = np.zeros((len(key),) + self.shape[1:])
        for i in range(len(key)):
            x[i,:] = self._get_bbox(p[i])
        return x

    def _get_bbox(self, p):
        if self.num_frames is None:
            return get_valid_bbox(p, relsize=self.relsize, square=self.square)
        else:
            b = np.zeros(self.shape[1:])
            for f in range(self.num_frames):
                b[f, :] = get_valid_bbox(p[f], self.relsize, self.square)
            return b

    def __len__(self):
        return len(self.poses)

    @property
    def shape(self):
        if self.num_frames is None:
            return (len(self), 4)
        else:
            return (len(self), self.num_frames, 4)

def get_valid_bbox(points, jprob=None, relsize=relsize_std, square=square_std):
    if jprob is None:
        v = get_valid_joints(points)
    else:
        v = np.squeeze(jprob > 0.5)

    if v.any():
        x = points[v==1, 0]
        y = points[v==1, 1]
    else:
        raise ValueError('get_valid_bbox: all points are invalid!')

    cx = (min(x) + max(x)) / 2.
    cy = (min(y) + max(y)) / 2.
    rw = (relsize * (max(x) - min(x))) / 2.
    rh = (relsize * (max(y) - min(y))) / 2.
    cy -= 0.02 * rh
    if square:
        rw = max(rw, rh)
        rh = max(rw, rh)

    return np.array([cx - rw, cy - rh, cx + rw, cy + rh])

def get_valid_bbox_array(pointarray, jprob=None, relsize=relsize_std,
        square=square_std):

    bboxes = np.zeros((len(pointarray), 4))
    v = None
    for i in range(len(pointarray)):
        if jprob is not None:
            v = jprob[i]
        bboxes[i, :] = get_valid_bbox(pointarray[i], jprob=v,
                relsize=relsize, square=square)

    return bboxes

def get_objpos_winsize(points, relsize=relsize_std, square=square_std):
    x = points[:, 0]
    y = points[:, 1]
    cx = (min(x) + max(x)) / 2.
    cy = (min(y) + max(y)) / 2.
    w = relsize * (max(x) - min(x))
    h = relsize * (max(y) - min(y))
    if square:
        w = max(w, h)
        h = max(w, h)

    return np.array([cx, cy]), (w, h)

def compute_grid_bboxes(frame_size, grid=(3, 2),
        relsize=relsize_std,
        square=square_std):

    bb_cnt = 0
    num_bb = 2 + grid[0]*grid[1]
    bboxes = np.zeros((num_bb, 4))

    def _smax(a, b):
        if square:
            return max(a, b), max(a, b)
        return a, b

    # Compute the first two bounding boxes as the full frame + relsize
    cx = frame_size[0] / 2
    cy = frame_size[1] / 2
    rw, rh = _smax(cx, cy)
    bboxes[bb_cnt, :] = np.array([cx-rw, cy-rh, cx+rw, cy+rh])
    bb_cnt += 1

    rw *= relsize
    rh *= relsize
    bboxes[bb_cnt, :] = np.array([cx-rw, cy-rh, cx+rw, cy+rh])
    bb_cnt += 1

    winrw = frame_size[0] / (grid[0]+1)
    winrh = frame_size[1] / (grid[1]+1)
    rw, rh = _smax(winrw, winrh)

    for j in range(1, grid[1]+1):
        for i in range(1, grid[0]+1):
            cx = i * winrw
            cy = j * winrh
            bboxes[bb_cnt, :] = np.array([cx-rw, cy-rh, cx+rw, cy+rh])
            bb_cnt += 1

    return bboxes

def bbox_to_objposwin(bbox, square=False):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    wx = bbox[2] - bbox[0]
    wy = bbox[3] - bbox[1]
    if square:
        wx = max(wx, wy)
        wy = wx

    return np.array([cx, cy]), np.array([wx, wy])

def objposwin_to_bbox(objpos, winsize):
    x1 = objpos[0] - winsize[0]/2
    y1 = objpos[1] - winsize[1]/2
    x2 = objpos[0] + winsize[0]/2
    y2 = objpos[1] + winsize[1]/2

    return np.array([x1, y1, x2, y2])


logkey_warn = set()
def get_gt_bbox(pose, visible, image_size, scale=1.0, logkey=None):
    assert len(pose.shape) == 3 and pose.shape[-1] >= 2, \
            'Invalid pose shape ({})'.format(pose.shape) \
            + ', expected (num_frames, num_joints, dim) vector'
    assert len(pose) == len(visible), \
            'pose and visible should have the same langth'

    if len(pose) == 1:
        idx = [0]
    else:
        idx = [0, int(len(pose)/2 + 0.5), len(pose)-1]

    clip_bbox = np.array([np.inf, np.inf, -np.inf, -np.inf])

    for i in idx:
        temp = pose[i, visible[i] >= 0.5]
        if len(temp) == 0:
            temp = pose[i, pose[i] > 0]

        if len(temp) > 0:
            b = get_valid_bbox(temp, relsize=1.5*scale)

            clip_bbox[0] = min(b[0], clip_bbox[0])
            clip_bbox[1] = min(b[1], clip_bbox[1])
            clip_bbox[2] = max(b[2], clip_bbox[2])
            clip_bbox[3] = max(b[3], clip_bbox[3])
        else:
            if logkey not in logkey_warn:
                warning('No ground-truth bounding box, ' \
                        'using full image (key {})!'.format(logkey))
            logkey_warn.add(logkey)

            clip_bbox[0] = min(0, clip_bbox[0])
            clip_bbox[1] = min(0, clip_bbox[1])
            clip_bbox[2] = max(image_size[0], clip_bbox[2])
            clip_bbox[3] = max(image_size[1], clip_bbox[3])

    return clip_bbox


def get_3d_crop_params(rootj, f, box_size_mm=2000):
    """Given a point in 3D (rootj), this function computes a 3D bounding box.

    # Arguments

        rootj: float array (n,3). 3D points in the center of the bounding boxes.
        f: float array (1,2). Camera focal length, in pixels.
        box_size_mm: float. The size of each bounding box, for individual
            points, in millimeters.

    # Return
        This function return a single bounding for a set of points. The final
        bouding box is the maximum among all individual bounding boxes, such as
        it has not necessarily the `box_size_mm`.

        objpos: (2,):(u,v) position in pixels.
        winsize: (2,):(w,h) size of the cropped region, in pixels.
        zrange: (2,):(min_z,max_z) range in z coordinates (for depth).
    """

    assert len(rootj.shape) == 2 and rootj.shape[-1] == 3, 'Invalid rootj ' \
            + 'shape ({}), expected (n, 3) vector'.format(rootj.shape)

    if len(rootj) < 3:
        idx = [i for i in range(len(rootj))]
    else:
        idx = [0, int(len(rootj)/2 + 0.5), len(rootj)-1]

    u1 = v1 = d1 = np.inf
    u2 = v2 = d2 = -np.inf
    rad = box_size_mm / 2.

    for i in idx:
        u, v, d = rootj[i, :]
        su = 1.125 * rad * f[0,0] / d
        sv = 1.125 * rad * f[0,1] / d
        u1 = min(u1, u - su)
        v1 = min(v1, v - sv)
        d1 = min(d1, d - rad)
        u2 = max(u2, u + su)
        v2 = max(v2, v + sv)
        d2 = max(d2, d + rad)

    objpos, winsize = bbox_to_objposwin([u1, v1, u2, v2], square=True)

    return objpos, winsize, np.array([d1, d2])


def auto_bbox_cropping(gt_pose_uvd, focal, box_size_mm, dconf, imgsize,
        bbox_crop_mode='3d',
        bbox_pred=None, # Used only on 2D mode and if pose_pred is not given
        pose_pred=None, # First option, if given
        mode=TRAIN_MODE,
        bbox_ratio_train=0.0):

    assert bbox_crop_mode in [None, '2d', '3d'], \
            'Invalid bbox_crop_mode `{}`'.format(bbox_crop_mode)

    if bbox_crop_mode == '3d':
        assert (gt_pose_uvd is not None) or (pose_pred is not None), (
                'In 3D bbox crop mode, `gt_pose_uvd` or `pose_pred` must be '
                'given')

    if mode == TRAIN_MODE:
        assert gt_pose_uvd is not None, \
                'In training mode, `gt_pose_uvd` is required'

    rad = dconf['scale'] * box_size_mm / 2.
    yshift = 0

    if pose_pred is not None:
        pose = pose_pred
    elif gt_pose_uvd is not None:
        pose = gt_pose_uvd
    else:
        pose = None

    if pose is not None:
        zrange = np.array([pose[0, 2] - rad, pose[0, 2] + rad])
    else:
        zrange = None

    def _crop_from_rootj(rootj):
        u, v, d = rootj
        su = 1.125 * rad * focal[0,0] / d
        sv = 1.125 * rad * focal[0,1] / d
        u1 = u - su
        v1 = v - sv
        d1 = d - rad
        u2 = u + su
        v2 = v + sv
        d2 = d + rad
        objpos, winsize = bbox_to_objposwin([u1, v1, u2, v2], square=True)

        return objpos, winsize, np.array([d1, d2])


    if (mode == TEST_MODE) or (mode == VALID_MODE):

        if bbox_crop_mode is None:
            objpos = np.array([imgsize[0], imgsize[1]]) / 2.
            winsize = imgsize

        elif bbox_crop_mode == '2d':
            if pose is not None:
                bbox = get_valid_bbox(pose[:, 0:2])
            else:
                bbox = bbox_pred

            objpos, winsize = bbox_to_objposwin(bbox, square=True)
            winsize[0] *= dconf['scale']
            winsize[1] *= dconf['scale']

        else: # '3d'
            objpos, winsize, zrange = _crop_from_rootj(pose[0])
            yshift = 256 // 11

    elif mode == TRAIN_MODE:
        objpos, winsize, zrange = _crop_from_rootj(pose[0])

        if bbox_pred is not None:
            bbox = bbox_pred
        else:
            bbox = get_valid_bbox(pose[:, 0:2])

        if bbox_crop_mode is None:
            objpos = np.array([imgsize[0], imgsize[1]]) / 2.
            winsize = imgsize

        elif bbox_crop_mode == '2d':
            objpos, winsize = bbox_to_objposwin(bbox, square=True)

            winsize[0] *= dconf['scale']
            winsize[1] *= dconf['scale']

        else: # '3d'
            """Use a 2d bbox with probability `bbox_ratio_train`"""
            if np.random.rand() <= bbox_ratio_train:
                objpos, winsize = bbox_to_objposwin(bbox, square=True)
            else:
                yshift = 256 // 11

    else:
        raise ValueError('Invalid mode `{}`'.format(mode))

    objpos += dconf['scale'] \
            * np.array([dconf['transx'], yshift + dconf['transy']])

    return objpos, winsize, zrange

