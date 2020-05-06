import numpy as np
from PIL import Image

from .io import WARNING
from .io import FAIL
from .io import printcn
from .io import warning

from .pose import pa16j2d
from .pose import pa17j3d
from .pose import pa20j3d
from .pose import pa21j3d
from .pose import coco17j
from .pose import dsl72j3d
from .pose import est34j3d
from .pose import dst68j3d
from .pose import h36m23j3d
from .pose import mpiinf3d28j3d
from .colors import cnames
from .colors import hex_colors
from .colors import hexcolor2tuple

# set to 'coco' to use COCO layout, 'pa' pose alternated if by default.
desambiguation_17j = 'pa'

try:
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14
    rcParams['legend.fontsize'] = 14
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['grid.alpha'] = 0.0

except Exception as e:
    warning(str(e), FAIL)
    plt = None

try:
    from mayavi import mlab

except Exception as e:
    warning(str(e), FAIL)
    mlab = None


def data_to_image(x, gray_scale=False):
    """ Convert 'x' to a RGB Image object.

    # Arguments
        x: image in the format (num_cols, num_rows, 3) for RGB images or
            (num_cols, num_rows) for gray scale images. If None, return a
            light gray image with size 100x100.
        gray_scale: convert the RGB color space to a RGB gray scale space.
    """

    if x is None:
        x = 224 * np.ones((100, 100, 3), dtype=np.uint8)

    if x.max() - x.min() > 0.:
        buf = 255. * (x - x.min()) / (x.max() - x.min())
    else:
        buf = x.copy()

    if len(buf.shape) == 3:
        (w, h) = buf.shape[0:2]
        num_ch = buf.shape[2]
    else:
        (h, w) = buf.shape
        num_ch = 1

    if ((num_ch is 3) and gray_scale):
        g = 0.2989*buf[:,:,0] + 0.5870*buf[:,:,1] + 0.1140*buf[:,:,2]
        buf[:,:,0] = g
        buf[:,:,1] = g
        buf[:,:,2] = g
    elif num_ch is 1:
        aux = np.zeros((h, w, 3), dtype=buf.dtype)
        aux[:,:,0] = buf
        aux[:,:,1] = buf
        aux[:,:,2] = buf
        buf = aux

    return Image.fromarray(buf.astype(np.uint8), 'RGB')


def show(x, gray_scale=False, jet_cmap=False, filename=None):
    """ Show 'x' as an image on the screen.
    """
    if jet_cmap is False:
        img = data_to_image(x, gray_scale=gray_scale)
    else:
        if plt is None:
            warning('pyplot not defined!')
            return
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=x.min(), vmax=x.max())
        img = cmap(norm(x))
    if filename:
        plt.imsave(filename, img)
    else:
        plt.imshow(img)
        plt.show()


def draw(x=None,
        skels=[],
        bboxes=[],
        bbox_color='g',
        abs_pos=False,
        plot3d=False,
        single_window=False,
        figsize=(16,9),
        axis='on',
        ticks='on',
        facecolor='white',
        azimuth=65,
        dpi=100,
        lw=4,
        filename=None):

    # Configure the ploting environment
    if plt is None:
        warning('pyplot not defined!')
        return

    # Plot 'x' and draw over it the skeletons and the bounding boxes
    img = data_to_image(x)
    if abs_pos:
        w = None
        h = None
    else:
        w,h = img.size

    def add_subimage(f, subplot, img):
        ax = f.add_subplot(subplot)
        plt.imshow(img, zorder=-1)
        return ax

    fig = [plt.figure(figsize=figsize)]
    ax = []

    if plot3d:
        if single_window:
            ax.append(add_subimage(fig[0], 121, img))
            ax.append(fig[0].add_subplot(122, projection='3d'))
        else:
            ax.append(add_subimage(fig[0], 111, img))
            fig.append(plt.figure(figsize=figsize))
            ax.append(fig[1].add_subplot(111, projection='3d'))
    else:
        ax.append(add_subimage(fig[0], 111, img))

    ax[0].axis('off') # Always grids off on RGB image

    plt.axis(axis)
    # for a in ax:
        # a.set_facecolor((0.95, 0.95, 0.95))

    # Plotting skeletons if not None
    if skels is not None:
        if isinstance(skels, list) or len(skels.shape) == 3:
            for s in skels:
                plot_skeleton_2d(ax[0], s, h=h, w=w, lw=lw)
                if plot3d:
                    plot_3d_pose(s, subplot=ax[-1], azimuth=azimuth, ticks=ticks)
        else:
            plot_skeleton_2d(ax[0], skels, h=h, w=w, lw=lw)
            if plot3d:
                plot_3d_pose(skels, subplot=ax[-1], azimuth=azimuth,
                        ticks=ticks)

    # Plotting bounding boxes if not None
    if bboxes is not None:
        if isinstance(bboxes, list) or bboxes.ndim == 2:
            for b, c in zip(bboxes, bbox_color):
                _plot_bbox(ax[0], b, h=h, w=w, c=c, lw=4)
        else:
            _plot_bbox(ax[0], bboxes, h=h, w=w, c=bbox_color, lw=4)


    if filename:
        fig[0].savefig(filename + '.png', bbox_inches='tight', pad_inches=0,
                facecolor=facecolor, dpi=dpi)
        if plot3d and (single_window is False):
            fig[-1].savefig(filename + '_3d.png',
                    bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

    for i in range(len(fig)):
        plt.close(fig[i])


def _get_poselayout(num_joints):
    if num_joints == 16:
        return pa16j2d.color, pa16j2d.cmap, pa16j2d.links
    elif num_joints == 17:
        if desambiguation_17j == 'coco':
            return coco17j.color, coco17j.cmap, coco17j.links
        else:
            return pa17j3d.color, pa17j3d.cmap, pa17j3d.links
    elif num_joints == 20:
        return pa20j3d.color, pa20j3d.cmap, pa20j3d.links
    elif num_joints == 21:
        return pa21j3d.color, pa21j3d.cmap, pa21j3d.links
    elif num_joints == 23:
        return h36m23j3d.color, h36m23j3d.cmap, h36m23j3d.links
    elif num_joints == 28:
        return mpiinf3d28j3d.color, mpiinf3d28j3d.cmap, mpiinf3d28j3d.links
    elif num_joints == 34:
        return est34j3d.color, est34j3d.cmap, est34j3d.links
    elif num_joints == 68:
        return dst68j3d.color, dst68j3d.cmap, dst68j3d.links
    elif num_joints == 72:
        return dsl72j3d.color, dsl72j3d.cmap, dsl72j3d.links


def plot_3d_pose(pose, subplot=None, filename=None, color=None, lw=3,
        azimuth=65, ticks='on'):

    if plt is None:
        raise Exception('"matplotlib" is required for 3D pose plotting!')

    num_joints, dim = pose.shape
    assert dim in [2, 3], 'Invalid pose dimension (%d)' % dim
    assert num_joints in [16, 17, 20, 21, 23, 28, 34, 68, 72], \
            'Unsupported number of joints (%d)' % num_joints

    col, cmap, links = _get_poselayout(num_joints)
    if color is None:
        color = col

    def _func_and(x):
        if x.all():
            return 1
        return 0

    points = np.zeros((num_joints, 3))
    for d in range(dim):
        points[:,d] = pose[:,d]
    # for i in range(num_joints):
        # points[i, 2] = max(0, points[i, 2])

    valid = np.apply_along_axis(_func_and, axis=1, arr=(points[:,0:2] > -1e6))

    if subplot is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = None
        ax = subplot

    for j in range(num_joints):
        if valid[j]:
            x, y, z = points[j]
            ax.scatter([x], [y], [z], lw=lw, c=color[cmap[j]], zorder=2)

    for i in links:
        if valid[i[0]] and valid[i[1]]:
            c = color[cmap[i[0]]]
            ax.plot(points[i, 0], points[i, 1], points[i, 2], c=c, lw=lw,
                    zorder=1)

    ax.view_init(10, azimuth)
    ax.set_aspect('equal')
    if ticks == 'on':
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


    rootj = points[0]
    xmin = rootj[0] -1000
    xmax = rootj[0] +1000
    ymin = rootj[1] -1000
    ymax = rootj[1] +1000
    zmax = 2000

    """Plot the ground surface."""
    # point  = np.array([0, 0, 0])
    # normal = np.array([0, 0, 1])
    # d = -point.dot(normal)
    # xx, yy = np.meshgrid(
            # range(int(xmin), int(xmax), 200),
            # range(int(ymin), int(ymax), 200))
    # z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    # ax.plot_surface(xx, yy, z, zorder=-1, shade=True, color='r')


    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    # ax.set_zlim([0, zmax])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    plt.gca().invert_xaxis()
    plt.gca().invert_zaxis()

    if fig is not None:
        if filename:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close(fig)


def _plot_bbox(subplot, bbox, h=None, w=None, scale=16, lw=2, c=None):
    assert len(bbox) == 4

    b = bbox.copy()
    if w is not None:
       b[0] *= (w - 1)
       b[2] *= (w - 1)
    if h is not None:
       b[1] *= (h - 1)
       b[3] *= (h - 1)

    if c is None:
        c = hex_colors[np.random.randint(len(hex_colors))]

    x = np.array([b[0], b[2], b[2], b[0], b[0]])
    y = np.array([b[1], b[1], b[3], b[3], b[1]])
    subplot.plot(x, y, lw=lw, c=c, zorder=1)


def plot_skeleton_2d(subplot, skel, h=None, w=None,
        joints=True, links=True, scale=16, lw=2):

    s = skel.copy()
    num_joints = len(s)
    assert num_joints in [16, 17, 20, 21, 23, 28, 34, 68, 72], \
            'Unsupported number of joints (%d)' % num_joints

    color, cmap, links = _get_poselayout(num_joints)

    x = s[:,0]
    y = s[:,1]
    v = s > -1e6
    v = v.any(axis=1).astype(np.float32)

    # Convert normalized skeletons to image coordinates.
    if w is not None:
        x *= (w - 1)
    if h is not None:
        y *= (h - 1)

    if joints:
        for i in range(len(v)):
            if v[i] > 0:
                c = color[cmap[i]]
                subplot.scatter(x=x[i], y=y[i], c=c, lw=lw, s=scale, zorder=2)

    if links:
        for i in links:
            if ((v[i[0]] > 0) and (v[i[1]] > 0)):
                c = color[cmap[i[0]]]
                subplot.plot(x[i], y[i], lw=lw, c=c, zorder=1)

def mplot3d(pose):
    assert mlab is not None, 'Required mayavi.mlab was not loaded correctly!'
    return


def mlab_plot_joints(pose, color=(0., 0., 1.), cmap=None, psize=None, scale=1):
    assert pose.ndim == 2, \
            'Expected a (N, dim+1) pose, got {}'.format(pose.shape)

    if isinstance(color, list):
        if cmap is not None:
            fc = lambda x: hexcolor2tuple(color[cmap[x]])
        else:
            fc = lambda x: color[x]
    else:
        fc = lambda x: color

    if psize is None:
        ps = len(pose) * [scale]
    else:
        ps = [scale * x for x in psize]

    for i in range(len(pose)):
        mlab.points3d(pose[i, 0], pose[i, 1], pose[i, 2], color=fc(i),
                scale_factor=ps[i], resolution=16)

def mlab_plot_links(pose, links, color=(1., 0., 0.), scale=1):
    assert pose.ndim == 2, \
            'Expected a (N, dim+1) pose, got {}'.format(pose.shape)

    if isinstance(color, list):
        fc = lambda x: hexcolor2tuple(color[links[x][0]])
    else:
        fc = lambda x: color
    fw = lambda x: links[x][1]
    fi = lambda x: links[x][2]
    fj = lambda x: links[x][3]

    for x in range(len(links)):
        mlab.plot3d(
                pose[[fi(x), fj(x)], 0],
                pose[[fi(x), fj(x)], 1],
                pose[[fi(x), fj(x)], 2],
                color=fc(x), tube_radius=fw(x) * scale, tube_sides=12)

def mlab_plot_pose(pose, links, color, cmap, psize, scale=1):
    mlab_plot_links(pose, links, color=color, scale=scale)
    mlab_plot_joints(pose, color=color, cmap=cmap, psize=psize, scale=scale)


