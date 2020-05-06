import os

import json
import numpy as np
from keras.callbacks import Callback

from .evaluation import eval_human36m_mm_error
from .evaluation import eval_human36m_mm_error_model
from .evaluation import eval_mpii_pckh
from .evaluation import eval_mpii3dhp_mm_error
from .utils import *

class SaveModel(Callback):

    def __init__(self, filepath, model_to_save=None, save_best_only=False,
            callback_to_monitor=None, verbose=1):

        if save_best_only and callback_to_monitor is None:
            warning('Cannot save the best model with no callback monitor')

        self.filepath = filepath
        self.model_to_save = model_to_save
        self.save_best_only = save_best_only
        self.callback_to_monitor = callback_to_monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        if self.model_to_save is not None:
            model = self.model_to_save
        else:
            model = self.model

        filename = self.filepath.format(epoch=epoch + 1)

        if self.best_epoch == epoch + 1 or not self.save_best_only:
            if self.verbose:
                printnl('Saving model @epoch=%05d to %s' \
                        % (epoch + 1, filename))
            model.save_weights(filename)

    @property
    def best_epoch(self):
        if self.callback_to_monitor is not None:
            return self.callback_to_monitor.best_epoch
        else:
            return None


class H36MEvalCallback(Callback):

    def __init__(self, x, aref, pose_w, afmat, scam, action, rootz=None,
            batch_size=24, eval_model=None, map_to_pa17j=None,
            eval_cam_pred=False, logdir=None):

        self.x = x
        self.aref = aref
        self.pose_w = pose_w
        self.afmat = afmat
        self.scam = scam
        self.action = action
        self.rootz = rootz
        self.batch_size = batch_size
        self.eval_model = eval_model
        self.map_to_pa17j = map_to_pa17j
        self.eval_cam_pred = eval_cam_pred
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        scores = eval_human36m_mm_error_model(model, self.x, self.aref,
                self.pose_w, self.afmat, self.scam, self.action,
                rootz=self.rootz,
                batch_size=self.batch_size,
                map_to_pa17j=self.map_to_pa17j,
                eval_cam_pred=self.eval_cam_pred)

        epoch += 1
        if self.logdir is not None:
            if not hasattr(self, 'logarray'):
                self.logarray = {}
            self.logarray[epoch] = scores
            with open(os.path.join(self.logdir, 'h36m_val.json'), 'w') as f:
                json.dump(self.logarray, f)

        cur_best = min(scores)
        self.scores[epoch] = cur_best

        printcn('Best score is %.1f at epoch %d' % \
                (self.best_score, self.best_epoch), OKBLUE)


    @property
    def best_epoch(self):
        if len(self.scores) > 0:
            # Get the key of the minimum value from a dict
            return min(self.scores, key=self.scores.get)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.scores) > 0:
            # Get the minimum value from a dict
            return self.scores[self.best_epoch]
        else:
            return np.inf


class MPIINF3DEvalCallback(Callback):

    def __init__(self, x, aref, pose_w, afmat, sub, rootz=None,
            resol_z=2000., batch_size=24, eval_model=None, map_to_pa17j=None,
            logdir=None):

        self.x = x
        self.aref = aref
        self.pose_w = pose_w
        self.afmat = afmat
        self.sub = sub
        self.rootz = rootz
        self.resol_z = resol_z
        self.batch_size = batch_size
        self.eval_model = eval_model
        self.map_to_pa17j = map_to_pa17j
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        scores = eval_mpii3dhp_mm_error(model, self.x, self.aref,
                self.pose_w, self.afmat, self.sub,
                rootz=self.rootz,
                batch_size=self.batch_size,
                map_to_pa17j=self.map_to_pa17j)

        epoch += 1
        if self.logdir is not None:
            if not hasattr(self, 'logarray'):
                self.logarray = {}
            self.logarray[epoch] = scores
            with open(os.path.join(self.logdir, 'mpiinf3d_val.json'), 'w') as f:
                json.dump(self.logarray, f)

        cur_best = min(scores)
        self.scores[epoch] = cur_best

        printcn('Best score is %.1f at epoch %d' % \
                (self.best_score, self.best_epoch), OKBLUE)


    @property
    def best_epoch(self):
        if len(self.scores) > 0:
            # Get the key of the minimum value from a dict
            return min(self.scores, key=self.scores.get)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.scores) > 0:
            # Get the minimum value from a dict
            return self.scores[self.best_epoch]
        else:
            return np.inf


class H36MEvalAnchoredCallback(Callback):

    def __init__(self, generator, anchors,
            workers=4,
            used_anchors=None,
            eval_model=None,
            map_to_pa17j=None,
            logdir=None):

        self.gen = generator
        self.anchors = anchors
        self.workers = workers
        self.used_anchors = used_anchors
        self.eval_model = eval_model
        self.map_to_pa17j = map_to_pa17j
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        pred, labels = predict_labels_generator(model, self.gen,
                workers=self.workers)

        absz = pred[0]
        pose = pred[-1]
        if self.used_anchors is not None:
            absz = absz[:, self.used_anchors]
            pose = pose[:, self.used_anchors]

        pose_w = labels[0]
        afmat = labels[1]
        scam = labels[3]
        action = labels[4]

        err = eval_human36m_mm_error(pose, absz, pose_w, afmat, scam,
                action, map_to_pa17j=self.map_to_pa17j)
        scores = [err]

        epoch += 1
        if self.logdir is not None:
            if not hasattr(self, 'logarray'):
                self.logarray = {}
            self.logarray[epoch] = scores
            with open(os.path.join(self.logdir, 'h36m_val.json'), 'w') as f:
                json.dump(self.logarray, f)

        cur_best = min(scores)
        self.scores[epoch] = cur_best

        printcn('Best score is %.1f at epoch %d' % \
                (self.best_score, self.best_epoch), OKBLUE)


    @property
    def best_epoch(self):
        if len(self.scores) > 0:
            # Get the key of the minimum value from a dict
            return min(self.scores, key=self.scores.get)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.scores) > 0:
            # Get the minimum value from a dict
            return self.scores[self.best_epoch]
        else:
            return np.inf


class MpiiEvalCallback(Callback):

    def __init__(self, fval, arefval, pval, afmat_val, headsize_val,
            batch_size=16, eval_model=None, map_to_mpii=None, logdir=None):

        self.fval = fval
        self.arefval = arefval
        self.pval = pval
        self.afmat_val = afmat_val
        self.headsize_val = headsize_val
        self.batch_size = batch_size
        self.eval_model = eval_model
        self.map_to_mpii = map_to_mpii
        self.scores = {}
        self.logdir = logdir

    def on_epoch_end(self, epoch, logs={}):
        if self.eval_model is not None:
            model = self.eval_model
        else:
            model = self.model

        scores = eval_mpii_pckh(model, self.fval, self.arefval, self.pval,
                self.afmat_val, self.headsize_val, batch_size=self.batch_size,
                map_to_mpii=self.map_to_mpii)

        epoch += 1
        if self.logdir is not None:
            if not hasattr(self, 'logarray'):
                self.logarray = {}
            self.logarray[epoch] = scores
            with open(os.path.join(self.logdir, 'mpii_val.json'), 'w') as f:
                json.dump(self.logarray, f)

        cur_best = max(scores)
        self.scores[epoch] = cur_best

        printcn('Best score is %.1f at epoch %d' % \
                (100*self.best_score, self.best_epoch), OKBLUE)


    @property
    def best_epoch(self):
        if len(self.scores) > 0:
            # Get the key of the maximum value from a dict
            return max(self.scores, key=self.scores.get)
        else:
            return np.inf

    @property
    def best_score(self):
        if len(self.scores) > 0:
            # Get the maximum value from a dict
            return self.scores[self.best_epoch]
        else:
            return 0
