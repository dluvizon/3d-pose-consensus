import numpy as np

CONF_MIN_VAL = 1e-4
CONF_MAX_VAL = 1. - CONF_MIN_VAL

from .colors import cnames

def pairflip(x):
    return x+1 if x % 2 == 0 else x-1

def pairlist(start, end, step=1):
    return [[x, x + step] for x in range(start, end, step)]

class mpii16j():
    """Defines the layout for MPII 2D Human Pose dataset. """
    joint_names = ['r_ankle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_ankle',
            'pelvis', 'thorax', 'neck', 'head',
            'r_wrist', 'r_elb', 'r_shoul', 'l_shoul', 'l_elb', 'l_wrist']
    num_joints = len(joint_names)
    dim = 2


class _pa16j():
    """Pose alternated with 16 joints (like Penn Action with three more
    joints on the spine.
    """
    num_joints = 16
    joint_names = ['pelvis', 'thorax', 'neck', 'head',
            'r_shoul', 'l_shoul', 'r_elb', 'l_elb', 'r_wrist', 'l_wrist',
            'r_hip', 'l_hip', 'r_knww', 'l_knee', 'r_ankle', 'l_ankle']

    """Horizontal flip mapping"""
    map_hflip = [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]

    """Projections from other layouts to the PA16J standard"""
    map_from_mpii = [6, 7, 8, 9, 12, 13, 11, 14, 10, 15, 2, 3, 1, 4, 0, 5]
    map_from_ntu = [0, 20, 2, 3, 4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]

    """Projections of PA16J to other formats"""
    map_to_pa13j = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    map_to_jhmdb = [2, 1, 3, 4, 5, 10, 11, 6, 7, 12, 13, 8, 9, 14, 15]
    map_to_mpii = [14, 12, 10, 11, 13, 15, 0, 1, 2, 3, 8, 6, 4, 5, 7, 9]
    map_to_lsp = [14, 12, 10, 11, 13, 15, 8, 6, 4, 5, 7, 9, 2, 3]

    """Color map"""
    color = ['g', 'r', 'b', 'y', 'm']
    cmap = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4]
    links = [[0, 1], [1, 2], [2, 3], [4, 6], [6, 8], [5, 7], [7, 9],
            [10, 12], [12, 14], [11, 13], [13, 15]]

class _pa17j():
    """Pose alternated with 17 joints (like _pa16j, with the middle spine).
    """
    num_joints = 17

    """Horizontal flip mapping"""
    map_hflip = [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 16]

    """Projections from other layouts to the PA17J standard"""
    map_from_h36m = \
            [0, 12, 13, 15, 25, 17, 26, 18, 27, 19, 1, 6, 2, 7, 3, 8, 11]
    map_from_ntu = _pa16j.map_from_ntu + [1]
    map_from_mpii3dhp = \
            [4, 5, 6, 7, 14, 9, 15, 10, 16, 11, 23, 18, 24, 19, 25, 20, 3]
    map_from_mpii3dhp_te = \
            [14, 1, 16, 0, 2, 5, 3, 6, 4, 7, 8, 11, 9, 12, 10, 13, 15]

    """Projections of PA17J to other formats"""
    map_to_pa13j = _pa16j.map_to_pa13j
    map_to_mpii = [14, 12, 10, 11, 13, 15, 0, 1, 2, 3, 8, 6, 4, 5, 7, 9]
    map_to_pa16j = list(range(16))

    """Color map"""
    color = ['g', 'r', 'b', 'y', 'm']
    color_mlab = [cnames['darkgreen'],
            cnames['crimson'],
            cnames['royalblue'],
            cnames['gold'],
            cnames['fuchsia'],
            ]
    cmap = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 0]
    links = [[0, 16], [16, 1], [1, 2], [2, 3], [4, 6], [6, 8], [5, 7], [7, 9],
            [10, 12], [12, 14], [11, 13], [13, 15]]
    mlab_links = [
            [0, 4, 0, 16],
            [0, 4, 16, 1],
            [0, 3, 1, 2],
            [0, 3, 2, 3],
            [1, 2, 4, 6],
            [1, 1, 6, 8],
            [2, 2, 5, 7],
            [2, 1, 7, 9],
            [3, 2, 10, 12],
            [3, 1, 12, 14],
            [4, 2, 11, 13],
            [4, 1, 13, 15]
            ]

    keypoint_size = [7, 7, 7, 7, 5, 5, 4, 4, 3, 3, 5, 5, 4, 4, 3, 3, 7]


class _pa20j():
    """Pose alternated with 20 joints. Similar to _pa16j, but with one more
    joint for hands and feet.
    """
    num_joints = 20

    """Horizontal flip mapping"""
    map_hflip = [0, 1, 2, 3, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16,
            19, 18]

    """Projections from other layouts to the PA20J standard"""
    map_from_h36m = [0, 12, 13, 15, 25, 17, 26, 18, 27, 19, 30, 22, 1, 6, 2,
            7, 3, 8, 4, 9]
    map_from_ntu = [0, 20, 2, 3, 4, 8, 5, 9, 6, 10, 7, 11, 12, 16, 13, 17, 14,
            18, 15, 19]

    """Projections of PA20J to other formats"""
    map_to_mpii = [16, 14, 12, 13, 15, 17, 0, 1, 2, 3, 8, 6, 4, 5, 7, 9]
    map_to_pa13j = [3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17]
    map_to_pa16j = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17]

    """Color map"""
    color = ['g', 'r', 'b', 'y', 'm']
    cmap = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4]
    links = [[0, 1], [1, 2], [2, 3], [4, 6], [6, 8], [8, 10], [5, 7], [7, 9],
            [9, 11], [12, 14], [14, 16], [16, 18], [13, 15], [15, 17], [17, 19]]

class _pa21j():
    """Pose alternated with 21 joints. Similar to _pa20j, but with one more
    joint referent to the 16th joint from _pa17j, for compatibility with H36M.
    """
    num_joints = 21

    """Horizontal flip mapping"""
    map_hflip = _pa20j.map_hflip + [20]

    """Projections from other layouts to the PA21J standard"""
    map_from_h36m = _pa20j.map_from_h36m + [11]
    map_from_ntu = _pa20j.map_from_ntu + [1]

    """Projections of PA20J to other formats"""
    map_to_mpii = _pa20j.map_to_mpii
    map_to_pa13j = _pa20j.map_to_pa13j
    map_to_pa16j = _pa20j.map_to_pa16j
    map_to_pa17j = _pa20j.map_to_pa16j + [20]

    """Color map"""
    color = ['g', 'r', 'b', 'y', 'm']
    cmap = [0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 0]
    links = [[0, 20], [20, 1], [1, 2], [2, 3], [4, 6], [6, 8], [8, 10], [5, 7],
            [7, 9], [9, 11], [12, 14], [14, 16], [16, 18], [13, 15], [15, 17],
            [17, 19]]

class coco17j():
    """Original layout for the MS COCO dataset."""
    num_joints = 17
    dim = 2

    """Horizontal flip mapping"""
    map_hflip = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

    """Color map"""
    color = ['g', 'r', 'b', 'y', 'm', 'w']
    cmap = [0, 0, 0, 5, 5, 0, 0, 2, 1, 2, 1, 0, 0, 4, 3, 4, 3]
    links = [[13, 15], [13, 11], [14, 16], [14, 12], [11, 12], [5, 11], [6,
        12], [5, 6], [7, 5], [8, 6], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
        [3, 1], [4, 2], [3, 5], [4, 6]]

class h36m23j3d():
    """Useful joints from Human36m. Only 23 joints from the 32 given are
    considered useful here. Not that we do not consider the `thumbs` as useful,
    since they are usually badly placed.
    """
    num_joints = 23
    dim = 3

    """Horizontal flip mapping"""
    map_hflip = list(range(5)) + \
            (np.array([pairflip(x) for x in range(4, 22)]) + 1).tolist()

    map_from_h36m = [0, 12, 13, 14, 15, 25, 17, 26, 18, 27, 19, 30, 22, 1, 6,
            2, 7, 3, 8, 4, 9, 5, 10]
    map_from_est34j = [0, 1, 5, 6, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33]
    map_to_pa17j = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 1]

    """Color map"""
    color = [cnames['darkgreen'],
            cnames['cyan'],
            cnames['crimson'],
            cnames['royalblue'],
            cnames['gold'],
            cnames['fuchsia'],
            ]
    cmap = 5*[0] + 4*[2, 3] + 5*[4, 5]
    keypoint_size = [9, 9, 9, 7, 7, 9, 9, 7, 7, 5, 5, 3, 3, 9, 9, 9, 9, 7, 7,
            5, 5, 3, 3]

    # links format: [color, width, idx1, idx2]
    links = [[0, 4, 0, 1], [0, 4, 1, 2], [0, 3, 2, 3], [0, 3, 3, 4],
            [1, 3, 13, 0], [1, 3, 0, 14], [1, 3, 13, 1], [1, 3, 14, 1],
            [1, 3, 1, 5], [1, 3, 1, 6], [1, 3, 5, 2], [1, 3, 2, 6],
            [2, 3, 5, 7], [2, 2, 7, 9], [2, 1, 9, 11],
            [3, 3, 6, 8], [3, 2, 8, 10], [3, 1, 10, 12],
            [4, 4, 13, 15], [4, 3, 15, 17], [4, 2, 17, 19], [4, 1, 19, 21],
            [5, 4, 14, 16], [5, 3, 16, 18], [5, 2, 18, 20], [5, 1, 20, 22]]


class mpiinf3d28j3d():
    """Body joints from MPI-INF-3DHP."""
    num_joints = 28
    dim = 3

    """Color map"""
    color = ['r']
    cmap = num_joints * [0]
    links = []

class est34j3d():
    """Elementary Skeleton Template."""
    num_joints = 34
    dim = 3

    """Projections to other formats."""
    map_to_mpii = [28, 26, 24, 25, 27, 29, 0, 4, 5, 9, 20, 18, 16, 17, 19, 21]
    map_to_mpii3dhp = [2, 3, 1, 1, 0, 5, 6, 9, 15, 17, 19, 21, 23,
            14, 16, 18, 20, 22, 25, 27, 29, 31, 33, 24, 26, 28, 30, 32]
    mat_to_mpi3d_te = [0, 5, 6, 9, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28,
            29, 1]
    map_to_h36m23j = [0, 1, 5, 6, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            26, 27, 28, 29, 30, 31, 32, 33]
    map_to_coco = \
            [7, 11, 10, 13, 12, 17, 16, 19, 18, 21, 20, 25, 24, 27, 26, 29, 28]
    map_to_pa16j = [0, 5, 6, 8, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29]
    map_to_pa17j = map_to_pa16j + [1]

    neighbors = np.array([
        [24, 25], # 0
        [0, 2], # 1
        [1, 4], # 2
        [1, 4], # 3
        [3, 5], # 4
        [4, 6], # 5
        [5, 7], # 6
        [5, 8], # 7
        [7, 9], # 8
        [8, 8], # 9
        [8, 7], # 10
        [8, 7], # 11
        [7, 10], # 12
        [7, 11], # 13
        [5, 16], # 14
        [5, 17], # 15
        [14, 18], # 16
        [15, 19], # 17
        [16, 20], # 18
        [17, 21], # 19
        [18, 22], # 20
        [19, 23], # 21
        [20, 20], # 22
        [21, 21], # 23
        [0, 26], # 24
        [0, 27], # 25
        [24, 28], # 26
        [25, 29], # 27
        [26, 30], # 28
        [27, 31], # 29
        [28, 32], # 30
        [29, 33], # 31
        [28, 30], # 32
        [29, 31], # 33
        ])

    """Horizontal flip mapping"""
    map_hflip = list(range(10)) + [pairflip(x) for x in range(10, 34)]

    """Skeleton or joint links"""
    color = [cnames['darkgreen'],
            cnames['cyan'],
            cnames['crimson'],
            cnames['royalblue'],
            cnames['gold'],
            cnames['fuchsia'],
            ]
    cmap = 5*[0] + 9*[1] + 5*[2, 3] + 5*[4, 5]
    links = pairlist(0, 4) + [[0, 24], [0, 25], [1, 24], [1, 25], [1, 16],
            [1, 17], [4, 16], [4, 17]] \
            + pairlist(5, 9) + [[7, 10], [7, 11], [10, 12], [11, 13]] \
            + pairlist(14, 22, 2) + pairlist(15, 23, 2) \
            + pairlist(24, 32, 2) + pairlist(25, 33, 2)
    newlinks = [[1, 24, 25], # 0
            [24, 25, 16, 17, 2], # 1
            [3], # 2
            [4], # 3
            [5, 16, 17], # 4
            [6, 14, 15], # 5
            [7, 12, 13], # 6
            [8, 10, 11], # 7
            [9], # 8
            [], # 9
            [], # 10
            [], # 11
            [9, 10], # 12
            [9, 11], # 13
            [], # 14
            [], # 15
            [14], # 16
            [15], # 17
            [16], # 18
            [17], # 19
            [18], # 20
            [19], # 21
            [20], # 22
            [21], # 23
            [], # 24
            [], # 25
            [24, 0], # 26
            [25, 0], # 27
            [26], # 28
            [27], # 29
            [28], # 30
            [29], # 31
            [30], # 32
            [31]] # 33
    keypoint_size = np.array(34*[1.])

class dst68j3d():
    """Dense Skeleton Template."""
    num_joints = 68
    dim = 3

    """Projections to other formats."""
    map_to_mpii = [62, 56, 48, 49, 57, 63, 0, 6, 7, 11, 32, 26, 20, 21, 27, 33]
    map_to_mpii3dhp = [4, 5, 3, 3, 0, 7, 8, 11, 16, 20, 26, 32, 36,
            17, 21, 27, 33, 37, 48, 56, 62, 64, 66, 49, 57, 63, 65, 67]
    map_to_h36m23j = [0, 3, 7, 8, 10, 20, 21, 26, 27, 32, 33, 36, 37,
            48, 49, 56, 57, 62, 63, 64, 65, 66, 67]
    map_to_coco = [9, 13, 12, 15, 14, 21, 20, 27, 26, 33, 32,
            49, 48, 57, 56, 63, 62]

    map_to_pa16j = [0, 7, 8, 10, 20, 21, 26, 27, 32, 33, 48, 49, 56, 57, 62, 63]
    map_to_pa17j = map_to_pa16j + [3]

    """Horizontal flip mapping"""
    map_hflip = list(range(12)) + [pairflip(x) for x in range(12, 68)]

    neighbors = np.array([
        [48, 49], # 0
        [0, 2], # 1
        [1, 3], # 2
        [2, 4], # 3
        [3, 5], # 4
        [4, 6], # 5
        [5, 7], # 6
        [6, 8], # 7
        [7, 9], # 8
        [8, 10], # 9
        [9, 11], # 10
        [9, 10], # 11
        [10, 14], # 12
        [10, 15], # 13
        [8, 12], # 14
        [8, 13], # 15
        [7, 18], # 16
        [7, 19], # 17
        [6, 20], # 18
        [6, 21], # 19
        [18, 22], # 20
        [19, 23], # 21
        [20, 24], # 22
        [21, 25], # 23
        [22, 26], # 24
        [23, 27], # 25
        [24, 28], # 26
        [25, 29], # 27
        [26, 30], # 28
        [27, 31], # 29
        [28, 32], # 30
        [29, 33], # 31
        [30, 34], # 32
        [31, 35], # 33
        [32, 36], # 34
        [33, 37], # 35
        [32, 34], # 36
        [33, 35], # 37
        [20, 40], # 38
        [21, 41], # 39
        [38, 3], # 40
        [39, 3], # 41
        [3, 40], # 42
        [3, 41], # 43
        [2, 3], # 44
        [2, 3], # 45
        [48, 1], # 46
        [49, 1], # 47
        [46, 50], # 48
        [47, 51], # 49
        [48, 54], # 50
        [49, 55], # 51
        [0, 48], # 52
        [0, 49], # 53
        [50, 56], # 54
        [51, 57], # 55
        [54, 58], # 56
        [55, 59], # 57
        [56, 60], # 58
        [57, 63], # 59
        [58, 62], # 60
        [59, 63], # 61
        [60, 64], # 62
        [61, 65], # 63
        [62, 66], # 64
        [63, 67], # 65
        [62, 64], # 66
        [63, 65], # 67
        ])

    softjoint_rules = [ # Additional points
            [1,  [0, 3],        [2/3, 1/3]],
            [2,  [0, 3],        [1/3, 2/3]],
            [18, [16, 20],      [1/2, 1/2]],
            [19, [17, 21],      [1/2, 1/2]],
            [22, [20, 26],      [2/3, 1/3]],
            [23, [21, 27],      [2/3, 1/3]],
            [24, [20, 26],      [1/3, 2/3]],
            [25, [21, 27],      [1/3, 2/3]],
            [28, [26, 32],      [2/3, 1/3]],
            [29, [27, 33],      [2/3, 1/3]],
            [30, [26, 32],      [1/3, 2/3]],
            [31, [27, 33],      [1/3, 2/3]],
            [34, [32, 36],      [1/2, 1/2]],
            [35, [33, 37],      [1/2, 1/2]],
            [38, [20, 3],       [2/3, 1/3]],
            [39, [21, 3],       [2/3, 1/3]],
            [40, [20, 3],       [1/3, 2/3]],
            [41, [21, 3],       [1/3, 2/3]],
            [42, [3, 20, 48],   [1/5, 2/5, 2/5]],
            [43, [3, 21, 49],   [1/5, 2/5, 2/5]],
            [44, [3, 48],       [2/3, 1/3]],
            [45, [3, 49],       [2/3, 1/3]],
            [46, [3, 48],       [1/3, 2/3]],
            [47, [3, 49],       [1/3, 2/3]],
            [50, [48, 56],      [2/3, 1/3]],
            [51, [49, 57],      [2/3, 1/3]],
            [52, [0, 56],       [2/3, 1/3]],
            [53, [0, 57],       [2/3, 1/3]],
            [54, [48, 56],      [1/3, 2/3]],
            [55, [49, 57],      [1/3, 2/3]],
            [58, [56, 62],      [2/3, 1/3]],
            [59, [57, 63],      [2/3, 1/3]],
            [60, [56, 62],      [1/3, 2/3]],
            [61, [57, 63],      [1/3, 2/3]],
            ]

    @staticmethod
    def compute_soft_joints(p, num_iter=1):
        assert (p.ndim == 2) and (p.shape == (dst68j3d.num_joints, 4)), (
                'Invalid pose, expected a pose + confidence '
                'tensor (%d,4), '
                'got %s' %(dst68j3d.num_joints, str(p.shape)))

        def _apply_rules(rules):
            for target, bases, weights in rules:
                try:
                    assert abs(sum(weights) - 1) < 1e-4
                    if np.isnan(p[target, 0:3]).all(): # joint not computed yet
                        for i in range(4):
                            p[target, i] = np.sum(p[bases, i] * weights)
                except:
                    warning('Invalid rule: {}, {}, {}'.format(
                        target, bases, weights))
                    raise

        for _ in range(num_iter):
            _apply_rules(dst68j3d.softjoint_rules)

        return p

    """Skeleton or joint links"""
    color = [cnames['darkgreen'],
            cnames['cyan'],
            cnames['crimson'],
            cnames['royalblue'],
            cnames['gold'],
            cnames['fuchsia'],
            ]
    cmap = 7*[0] + 9*[1] + 11*[2, 3] + 10*[0] + 10*[4, 5]
    links = pairlist(0, 7) + [[0, 49], [49, 47], [47, 45], [45, 3], [3, 40],
            [40, 38], [38, 20], [20, 6], [6, 21], [21, 39], [39, 41], [41, 3],
            [3, 44], [44, 46], [46, 48], [0, 48], [38, 42], [42, 46], [39, 43],
            [43, 47], [52, 0], [52, 56], [53, 0], [53, 57], [50, 48], [51, 49],
            [50, 54], [51, 55]] \
            + pairlist(16, 36, 2) + pairlist(17, 37, 2) \
            + pairlist(54, 66, 2) + pairlist(55, 67, 2)


class dsl72j3d():
    pass


class pa16j2d(_pa16j):
    dim = 2

class pa16j3d(_pa16j):
    dim = 3

class pa17j2d(_pa17j):
    dim = 2

class pa17j3d(_pa17j):
    dim = 3

class pa20j3d(_pa20j):
    dim = 3

class pa21j3d(_pa21j):
    dim = 3

class ntu25j3d():
    num_joints = 25
    dim = 3


def _func_and(x):
    if x.all():
        return 1
    return 0

def get_visible_joints(x, min_val=CONF_MIN_VAL, max_val=CONF_MAX_VAL,
        margin=0.0):

    assert x.ndim == 2, 'Invalid input shape {}'.format(x.shape)

    vnan = np.isnan(x)
    vis = (max_val - min_val) * np.prod((x > margin) * (x < 1 - margin),
            axis=-1) + min_val
    vis[np.sum(vnan, axis=-1, dtype=bool)] = np.nan

    return vis

def get_valid_joints(x):
    assert x.ndim == 2, 'Invalid input shape {}'.format(x.shape)

    vnan = np.isnan(x)
    x[vnan] = 0
    val = np.prod(x > -1e6, axis=-1)
    idx = np.sum(vnan, axis=-1, dtype=bool)
    if idx.any():
        val[idx] = np.nan

    return val


def assign_nn_confidence(c, ngb, num_iter=1):

    assert c.ndim in [2, 3] and ngb.ndim == 2 and c.shape[-2] == ngb.shape[0], \
            'Invalid confidence {} / neighbors {} shape'.format(
                    c.shape, ngb.shape)

    for _ in range(num_iter):
        if c.ndim == 2:
            newc = np.nanmean(c[ngb], axis=1)
        else:
            newc = np.nanmean(c[:, ngb], axis=2)
        c = np.where(np.isnan(c), newc, c)

    return c

