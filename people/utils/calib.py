import numpy as np

from .metrics import abs_mpjpe

def project_poses(poses, P):
    """Compute projected poses x = Pp."""

    assert poses.ndim == 2 and poses.shape[-1] == 3, \
            'Invalid pose dim at ext_proj {}'.format(poses.shape)
    assert P.shape == (3, 4), 'Invalid projection shape {}'.format(P.shape)

    p = np.concatenate([poses, np.ones((len(poses), 1))], axis=-1)
    x = np.matmul(P, p.T)

    return x.T

def predict_xy_mm(pred, f, c):
    num_samples, dim = pred.shape
    assert dim == 3, 'Invalid pose dim at predict_xy_mm ({})'.format(dim)
    assert f.shape == (1, 2), 'Invalid focal length matrix {}, expected (1,2)'.format(f.shape)
    assert c.shape == (1, 2), 'Invalid camera center matrix {}, expected (1,2)'.format(f.shape)

    uv = pred[:, 0:2]
    zz = np.tile(pred[:, 2:3], (1, 2))
    xy = zz * (uv - c) / f

    p = pred.copy()
    p[:, 0:2] = xy

    return p

def predict_uv_px(pc, f, c):
    num_samples, num_joints, dim = pc.shape
    assert dim == 3, 'Invalid pose dim at predict_uv_px ({})'.format(dim)
    assert f.shape == (1, 2), 'Invalid focal length matrix {}, expected (1,2)'.format(f.shape)
    assert c.shape == (1, 2), 'Invalid camera center matrix {}, expected (1,2)'.format(f.shape)

    xy = np.reshape(pc[:, :, 0:2], (num_samples * num_joints, 2))
    zz = np.tile(np.reshape(pc[:, :, 2], (num_samples * num_joints, 1)), (1, 2))
    uv = f * xy / zz + c

    p = pc.copy()
    p[:, :, 0:2] = np.reshape(uv, (num_samples, num_joints, 2))

    return p

def camera_inv_proj(pred, fx, fy, cx, cy):
    u, v, z = pred.T
    x = z * (u - cx) / fx
    y = z * (v - cy) / fy
    return np.array([x, y, z]).T

def get_idxs_sequence(meta, asec):
    idxs = []
    for i , m in enumerate(meta):
        if (m[0], m[1], m[2], m[3]) == asec:
            idxs.append(i)

    return idxs


def getP(X1, X2):
    n = np.shape(X1)[1]
    C = np.zeros((12, 12))
    k_x2 = np.zeros((12,1))
    # estimate both matrix
    for i, x in enumerate(X1):
        # put x1 as [[x y z 1 0 0 0 0 0 0 0 0], [0 0 0 0 x y z 1 0 0 0 0], [0 0 0 0 0 0 0 0 x y z 1]]
        x_k = np.kron(x, np.eye(3)).T
        # put x2 as [x; y; z]
        x2 = np.reshape(X2[i], (3,1))
        # accumulate statistics over x1.T * x2
        k_x2 += np.matmul(x_k, x2)
        # accumulate statistics over X1.T * x1
        C += np.matmul(x_k, x_k.T)
    # pseudo-inverse method
    P = np.matmul(np.linalg.inv(C), k_x2)
    # reshape to original form
    P = np.reshape(P, (4, 3)).T
    return P


def kabsch_alignment(X1, X2):
    """Compute a rotation matrix R (3, 3) that minimizes || R X1 - X2 ||F ."""

    assert X1.shape == X2.shape, \
            'Incompatible pose shapes {},{}'.format(X1.shape, X2.shape)
    assert X1.ndim == 2 and X1.shape[-1] == 3, \
            'Invalid pose dim at compute_extrinsic_param ({})'.format(dim)

    H = np.matmul(X2.T, X1)
    U, S, Vh = np.linalg.svd(H, full_matrices=True)
    d = np.linalg.det(np.matmul(Vh.T, U.T))

    A = np.eye(3)
    A[2,2] = d

    R = np.matmul(np.matmul(Vh.T, A), U.T).T

    return R


def optimize_translation(X1, X2, R12):

    assert X1.shape == X2.shape, \
            'Incompatible pose shapes {},{}'.format(X1.shape, X2.shape)
    assert X1.ndim == 2 and X1.shape[-1] == 3, \
            'Invalid pose dim at compute_extrinsic_param ({})'.format(dim)

    A = X1.T - np.matmul(R12.T, X2.T)
    t = np.mean(A, axis=-1, keepdims=True)

    return t

def optimize_focal(pred1, pc2, P21, c1):

    assert pred1.shape == pc2.shape, \
            'Invalid pred shape {} {}'.format(pred1.shape, pc2.shape)
    assert pred1.ndim == 2 and pred1.shape[-1] == 3, \
            'Invalid pred at optimize_focal {}'.format(pred1.shape)
    assert P21.shape == (3, 4), 'Invalid projection shape {}'.format(P21.shape)
    assert c1.shape == (1,2), 'Invalid c1 shape {}'.format(c1.shape)

    uv = pred1[:, 0:2]
    zz = np.tile(pred1[:,2:3], (1, 2))
    w1 = (uv - c1) * zz
    w1x = w1[:, 0:1].T
    w1y = w1[:, 1:2].T

    w2 = project_poses(pc2, P21)
    w2x = w2[:, 0:1].T
    w2y = w2[:, 1:2].T

    A = np.matmul(w1x, w1x.T)
    B = np.matmul(w2x, w1x.T)
    f1x_prime = np.matmul(np.linalg.inv(A), B)[0,0]

    A = np.matmul(w1y, w1y.T)
    B = np.matmul(w2y, w1y.T)
    f1y_prime = np.matmul(B, np.linalg.inv(A))[0,0]

    return np.array([[1. / f1x_prime, 1. / f1y_prime]])

def optimize_center(pred1, pc2, P21, f1):

    assert pred1.shape == pc2.shape, \
            'Invalid pred shape {} {}'.format(pred1.shape, pc2.shape)
    assert pred1.ndim == 2 and pred1.shape[-1] == 3, \
            'Invalid pred at optimize_focal {}'.format(pred1.shape)
    assert P21.shape == (3, 4), 'Invalid projection shape {}'.format(P21.shape)
    assert f1.shape == (1,2), 'Invalid f1 shape {}'.format(f1.shape)

    uv = pred1[:, 0:2]
    zz = np.tile(pred1[:,2:3], (1, 2))

    w2 = project_poses(pc2, P21)
    w2 = w2[:, 0:2]
    w2x = w2[:, 0:1].T
    w2y = w2[:, 1:2].T

    B = zz / f1
    Bx = B[:, 0:1].T
    By = B[:, 1:2].T

    A = uv * B - w2
    Ax = A[:, 0:1].T
    Ay = A[:, 1:2].T

    P = np.matmul(Bx, Bx.T)
    Q = np.matmul(Ax, Bx.T)
    c1x = np.matmul(Q, np.linalg.inv(P))[0,0]

    P = np.matmul(By, By.T)
    Q = np.matmul(Ay, By.T)
    c1y = np.matmul(Q, np.linalg.inv(P))[0,0]

    return np.array([[c1x, c1y]])

def std_mpjpe(p1, p2):
    return np.std(np.sqrt(np.sum(np.square(p1 - p2), axis=-1)))

def dist_mpjpe(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))

def predict_camera_parameters(puvd1, puvd2, miniter=100, maxiter=500,
        alpha=0.8, f_init=500, c_init=500):

    assert puvd1.shape == puvd2.shape, \
        'Incompatible pose shapes {},{}'.format(puvd1.shape, puvd2.shape)
    
    # Initialize variables
    f1 = np.array([[f_init, f_init]])
    f2 = np.array([[f_init, f_init]])
    c1 = np.array([[c_init, c_init]])
    c2 = np.array([[c_init, c_init]])

    pc1 = predict_xy_mm(puvd1, f1, c1)
    pc2 = predict_xy_mm(puvd2, f2, c2)

    X1 = pc1
    X2 = np.concatenate([pc2, np.ones((len(pc2), 1))], axis=-1)
    P21_init = getP(X2, X1)

    pc1_init = project_poses(pc2, P21_init)
    t2 = -np.matmul(np.linalg.inv(P21_init[:,:3]), P21_init[:,3:4])
    loss = []
    prev_loss = 1e4
    min_loss = 1e4
    
    spl = range(len(puvd1))

    for i in range(maxiter):
        puvd1_b = puvd1[spl]
        puvd2_b = puvd2[spl]
        pc1_b = pc1[spl]
        pc2_b = pc2[spl]
        
        R21 = kabsch_alignment(pc2_b - t2.T, pc1_b)
        t2 = optimize_translation(pc2_b, pc1_b, R21)
        R12 = np.linalg.inv(R21)
        t1 = -np.matmul(R21, t2)

        P21 = np.concatenate([R21, t1], axis=-1)
        P12 = np.concatenate([R12, t2], axis=-1)

        if i % 4 == 0:
            f1 = optimize_focal(puvd1_b, pc2_b, P21, c1)
            f1[0,0] = alpha*f1[0,0] + (1-alpha)*f1[0,1]
            f1[0,1] = (1-alpha)*f1[0,0] + alpha*f1[0,1]

        elif i % 4 == 1:
            f2 = optimize_focal(puvd2_b, pc1_b, P12, c2)
            f2[0,0] = alpha*f2[0,0] + (1-alpha)*f2[0,1]
            f2[0,1] = (1-alpha)*f2[0,0] + alpha*f2[0,1]

        elif i % 4 == 2:
            c1 = optimize_center(puvd1_b, pc2_b, P21, f1)
            c1[0,0] = alpha*c1[0,0] + (1-alpha)*c1[0,1]
            c1[0,1] = (1-alpha)*c1[0,0] + alpha*c1[0,1]

        elif i % 4 == 3:
            c2 = optimize_center(puvd2_b, pc1_b, P12, f2)
            c2[0,0] = alpha*c2[0,0] + (1-alpha)*c2[0,1]
            c2[0,1] = (1-alpha)*c2[0,0] + alpha*c2[0,1]
        
        pc1 = predict_xy_mm(puvd1, f1, c1)
        pc2 = predict_xy_mm(puvd2, f2, c2)

        pc2c1 = project_poses(pc2, P21)
        err = abs_mpjpe(pc2c1, pc1)
        std = std_mpjpe(pc2c1, pc1)
        dist = dist_mpjpe(pc2c1, pc1)
        spl = dist < err + 2*std
        loss.append(err)

        if i > miniter and (loss[-1] > prev_loss or loss[-1] > min_loss + 1):
            break

        prev_loss = 0.5*prev_loss + 0.5*loss[-1]
        if loss[-1] < min_loss:
            min_loss = loss[-1]
            f1o = f1.copy()
            c1o = c1.copy()
            f2o = f2.copy()
            c2o = c2.copy()
            P21o = P21.copy()
            P12o = P12.copy()

    return f1o, c1o, f2o, c2o, P21o, P12o, np.array(loss)

