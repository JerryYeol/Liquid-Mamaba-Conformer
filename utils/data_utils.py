# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : data_utils.py
import os
import random
from functools import lru_cache

import numpy as np
import pandas as pd
import pywt

from utils.alg_utils import EA

# meta.csv and y.npy live next to X.npy, same relative-path convention as
# data_process() in utils/dataloader.py
EEGST_DIR = './data/EEG-ST'
UPEEGST_DIR = './data/UP-EEGST'
LUEEGMI_DIR = './data/LU-EEGMI'


@lru_cache(maxsize=None)
def _eegst_subject_table():
    """(n, n0, n1) for each EEG-ST subject, in X.npy row order.

    AutoReject drops the trials it cannot repair, and it drops a different
    number for each subject -- 24 to 40 of the original 40, with 20 of the 46
    subjects losing nothing.  build_eegst.py writes meta.csv subject-major, so
    grouping in file order reproduces the row order of X.npy.

    Cached because the runners call this once per seed per subject and the
    files do not change while they run.
    """
    meta = pd.read_csv(os.path.join(EEGST_DIR, 'meta.csv'))
    y = np.load(os.path.join(EEGST_DIR, 'y.npy'))
    # Mask on the positional subject column, never on the DataFrame index:
    # y is a bare array, so an index that is not a RangeIndex (index_col=...)
    # would line the wrong labels up with the wrong rows and swap n0 with n1
    # without raising.  dict.fromkeys keeps first-appearance order, which is
    # the file order, which is the row order of X.npy.
    subject = meta['subject'].values
    table = []
    for s in dict.fromkeys(subject.tolist()):
        yy = y[subject == s]
        table.append((len(yy), int((yy == 0).sum()), int((yy == 1).sum())))
    return tuple(table)


def upeegst_dir(dataset):
    """Folder holding X.npy / meta.csv for 'UP-EEGST' or 'UP-EEGST-sesNN'.

    Same convention as LU-EEGMI, one step removed: the suffix after the dash
    picks the subfolder, so 'UP-EEGST-ses07' is data/UP-EEGST/ses-07/, while
    plain 'UP-EEGST' is data/UP-EEGST/ itself and pools all 278 sessions.  The
    folder keeps the dash the data_name drops -- data_name suffixes read
    'ses07' and folders are named 'ses-07' (data/UP-EEGST/split_by_session.py),
    so the two cannot be used interchangeably.
    """
    if dataset == 'UP-EEGST':
        return UPEEGST_DIR
    ses = dataset[len('UP-EEGST-'):]                 # 'ses07'
    return os.path.join(UPEEGST_DIR, 'ses-' + ses[len('ses'):])   # 'ses-07'


@lru_cache(maxsize=None)
def _upeegst_subject_table(dataset):
    """(n, n0, n1) for each UP-EEGST subject, in X.npy row order.

    preprocess_upeegst.py (the pooled folder) and split_by_session.py (one
    folder per session day) both write meta.csv subject-major, which is also
    the row order of X.npy, so grouping in file order gives the subjects back.
    Unlike EEG-ST, nothing is dropped here: UP-EEGST is ragged because its
    sessions are -- 17-26 per subject, 20-160 trials each -- not because a
    rejection step culled some, and a per-session folder inherits that
    property inside itself (20-160 trials per subject).

    Cached for the same reason as the EEG-ST table: the runners call this once
    per seed per subject.
    """
    root = upeegst_dir(dataset)
    meta = pd.read_csv(os.path.join(root, 'meta.csv'))
    y = np.load(os.path.join(root, 'y.npy'))
    # Positional column, same reason as _eegst_subject_table: meta.csv has no
    # meaningful index, so indexing into it by label would silently misalign.
    subject = meta['subject'].values
    table = []
    for s in dict.fromkeys(subject.tolist()):
        yy = y[subject == s]
        table.append((len(yy), int((yy == 0).sum()), int((yy == 1).sum())))
    return tuple(table)


@lru_cache(maxsize=None)
def _lueegmi_subject_table(dataset):
    """(n, n0, n1) for each LU-EEGMI subject, in X.npy row order.

    LU-EEGMI is one folder per session -- pre, post, follow -- and `dataset` is
    'LU-EEGMI-pre' and friends.  Trials per subject are ragged inside every one
    of the folders (79-100 in pre, 20-100 in follow) because the subjects ran
    different numbers of runs; preprocess_official.py writes meta.csv
    subject-major, so grouping in file order gives the subjects back.

    post/ and follow/ hold 23 subjects rather than 27: sub-03/06/08/11 never
    had those visits.  That is why num_subjects differs per folder in
    utils/dataloader.py, and why this table is the thing that knows the truth.

    Cached for the same reason as the other two tables: the runners call this
    once per seed per subject.
    """
    root = os.path.join(LUEEGMI_DIR, dataset[len('LU-EEGMI-'):])
    meta = pd.read_csv(os.path.join(root, 'meta.csv'))
    y = np.load(os.path.join(root, 'y.npy'))
    subject = meta['subject'].values
    table = []
    for s in dict.fromkeys(subject.tolist()):
        yy = y[subject == s]
        table.append((len(yy), int((yy == 0).sum()), int((yy == 1).sum())))
    return tuple(table)


def subject_trial_counts(dataset):
    """Trials per subject, in X.npy row order.  None when they are uniform.

    Every dataset but EEG-ST, UP-EEGST and the three LU-EEGMI folders keeps the
    same number of trials for every subject, so callers there go on dividing
    the row count evenly.  Those four arrive ragged -- EEG-ST out of AutoReject,
    UP-EEGST and LU-EEGMI because the recordings themselves are uneven -- so
    they have to be asked instead.
    """
    if dataset == 'EEG-ST':
        return tuple(n for n, _, _ in _eegst_subject_table())
    if dataset.startswith('UP-EEGST'):
        return tuple(n for n, _, _ in _upeegst_subject_table(dataset))
    if dataset.startswith('LU-EEGMI-'):
        return tuple(n for n, _, _ in _lueegmi_subject_table(dataset))
    return None


def subject_minority_count(dataset, idt):
    """Trials in subject `idt`'s smaller class.

    The ceiling on how many of each class a within-subject split can hold
    out for training without emptying the test set.  Only the ragged datasets
    need this -- anywhere else the answer is trial_num / 2.
    """
    assert subject_trial_counts(dataset) is not None, \
        '{} is not ragged; use trial_num / 2 instead'.format(dataset)
    if dataset == 'EEG-ST':
        table = _eegst_subject_table()
    elif dataset.startswith('UP-EEGST'):
        table = _upeegst_subject_table(dataset)
    else:
        table = _lueegmi_subject_table(dataset)
    return min(table[idt][1:])


@lru_cache(maxsize=None)
def _upeegst_session_table(dataset):
    """(n_trials, subject) for each (subject, session) block, in X.npy row order.

    preprocess_upeegst.py writes meta.csv subject-major and session-minor,
    which is also the row order of X.npy, so a session change is exactly a
    block boundary and reading the runs off in file order gives the blocks
    back with no offset arithmetic.  Blocks run 20-160 trials, far past what
    an 11 x 11 covariance needs.

    A per-session folder has one session in it, so its blocks are its subjects
    and this reduces to the subject table -- which is what data_loader wants
    there, a reference per subject.
    """
    meta = pd.read_csv(os.path.join(upeegst_dir(dataset), 'meta.csv'))
    # Positional columns, same reason as _eegst_subject_table: meta.csv has no
    # meaningful index, so indexing into it by label would silently misalign.
    subject = meta['subject'].values
    session = meta['session'].values
    # The subject column holds the original sub-XX numbers -- 1, 4, 5, ... --
    # while the runners address subjects by their 0..N-1 position in X.npy.
    # Reporting the raw label here would make data_loader drop the wrong
    # subject's blocks.  dict.fromkeys gives first-appearance order, i.e.
    # the position.
    position = {s: i for i, s in enumerate(dict.fromkeys(subject.tolist()))}
    table = []
    start = 0
    for i in range(1, len(meta) + 1):
        if i == len(meta) or subject[i] != subject[start] or session[i] != session[start]:
            table.append((i - start, position[int(subject[start])]))
            start = i
    return tuple(table)


def session_blocks(dataset):
    """Per-session block sizes and the subject each one belongs to, or None.

    UP-EEGST is the one dataset whose sessions disagree on scale: inside a
    subject the quietest session is two orders of magnitude smaller than the
    loudest.  The alignment (EA or SPDA, see align_fn) builds its reference out
    of whatever it is handed, so aligning a subject as a single block pools its
    loud and quiet sessions into one reference and leaves the contrast between
    them standing.  Cutting at session boundaries normalises across sessions
    and across subjects in the same step.

    Everywhere else the answer is None and callers keep their subject-wise
    split.
    """
    if not dataset.startswith('UP-EEGST'):
        return None
    counts, owners = zip(*_upeegst_session_table(dataset))
    return list(counts), list(owners)


def split_by_subject(dataset, X, y, num_subjects):
    """Per-subject chunks of X and y, in subject order.

    Ragged datasets are cut at their cumulative trial counts -- the same
    cumulative-index form traintest_split_cross_subject_trialdiffnum already
    uses for Zhou2016.  Passed as a plain int, np.split divides the rows
    evenly exactly as it always did, so every other dataset is untouched.
    """
    counts = subject_trial_counts(dataset)
    if counts is None:
        bounds = num_subjects
    else:
        assert len(counts) == num_subjects, \
            '{}: {} subjects in meta.csv, expected {}'.format(
                dataset, len(counts), num_subjects)
        bounds = np.cumsum(counts)[:-1]
    return (np.split(X, indices_or_sections=bounds, axis=0),
            np.split(y, indices_or_sections=bounds, axis=0))


def split_data(data, axis, times):
    # Splitting data into multiple sections. data: (trials, channels, time_samples)
    data_split = np.split(data, indices_or_sections=times, axis=axis)
    return data_split


def convert_label(labels, axis, threshold):
    # Converting labels to 0 or 1, based on a certain threshold
    label_01 = np.where(labels > threshold, 1, 0)
    print(label_01)
    return label_01


def time_cut(data, cut_percentage):
    # Time Cutting: cut at a certain percentage of the time. data: (..., ..., time_samples)
    data = data[:, :, :int(data.shape[2] * cut_percentage)]
    return data


def traintest_split_cross_subject(dataset, X, y, num_subjects, test_subject_id):
    data_subjects, labels_subjects = split_by_subject(dataset, X, y, num_subjects)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y

def traintest_split_cross_subject_dwt(args, X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    l_id = [i for i in range(train_y.shape[0]) if train_y[i] == 0]
    r_id = [i for i in range(train_y.shape[0]) if train_y[i] == 1]
    Xs0 = train_x[l_id]
    ys0 = train_y[l_id]
    Xs1 = train_x[r_id]
    ys1 = train_y[r_id]

    l_index = [i for i in range(test_y.shape[0]) if test_y[i] == 0]
    r_index = [i for i in range(test_y.shape[0]) if test_y[i] == 1]
    X_la = test_x[l_index, :, :]
    X_ra = test_x[r_index, :, :]
    y_la = test_y[l_index]
    y_ra = test_y[r_index]
    nsamples = 1
    # target train data
    X_lat = X_la[:nsamples, :, :]
    X_rat = X_ra[:nsamples, :, :]
    y_lat = y_la[:nsamples]
    y_rat = y_ra[:nsamples]
    # target test data
    X_lae = X_la[nsamples:, :, :]
    X_rae = X_ra[nsamples:, :, :]
    y_lae = y_la[nsamples:]
    y_rae = y_ra[nsamples:]
    X_tar_e = np.concatenate((X_lae, X_rae), axis=0)
    y_tar_e = np.concatenate((y_lae, y_rae), axis=0)

    if args.DWT_aug:
        src_aug_x = []
        src_aug_y = []
        wavename = 'db5'  # TODO db5
        TcA0, TcD0 = pywt.dwt(X_lat, wavename)
        TcA1, TcD1 = pywt.dwt(X_rat, wavename)
        for k in range(Xs0.shape[0]):
            Xss0 = Xs0[k, :, :]
            Xss0 = np.expand_dims(Xss0, axis=0)
            ScA0, ScD0 = pywt.dwt(Xss0, wavename)  # TODO dwt
            Xs_aug0 = pywt.idwt(ScA0, TcD0, 'db5', 'smooth')
            Xt_aug0 = pywt.idwt(TcA0, ScD0, 'db5', 'smooth')
            # if args.align:
            #     Xs_aug0 = EA(Xs_aug0)
            #     Xt_aug0 = EA(Xt_aug0)
            src_aug_x.append(Xs_aug0[:, :, :Xss0.shape[-1]])
            src_aug_x.append(Xt_aug0[:, :, :Xss0.shape[-1]])
            src_aug_y.append(y_lat)
            src_aug_y.append(y_lat)
        for m in range(Xs1.shape[0]):
            Xss1 = Xs1[m, :, :]
            Xss1 = np.expand_dims(Xss1, axis=0)
            ScA1, ScD1 = pywt.dwt(Xss1, wavename)  # TODO dwt
            Xs_aug1 = pywt.idwt(ScA1, TcD1, 'db5', 'smooth')
            Xt_aug1 = pywt.idwt(TcA1, ScD1, 'db5', 'smooth')
            # if args.align:
            #     Xs_aug1 = EA(Xs_aug1)
            #     Xt_aug1 = EA(Xt_aug1)
            src_aug_x.append(Xs_aug1[:, :, :Xss1.shape[-1]])
            src_aug_x.append(Xt_aug1[:, :, :Xss1.shape[-1]])
            src_aug_y.append(y_rat)
            src_aug_y.append(y_rat)
        src_aug_x = np.concatenate(src_aug_x)
        src_aug_y = np.concatenate(src_aug_y)
        src_x = np.concatenate((Xs0, Xs0, src_aug_x))
        src_y = np.concatenate((ys0, ys1, src_aug_y))
        return src_x, src_y, test_x, test_y


def traintest_split_cross_subject_dwt_class(args, X, y, num_subjects, test_subject_id):
    src_x = []
    src_y = []
    src_x_0 = []
    src_y_0 = []
    src_x_1 = []
    src_y_1 = []
    for k in range(num_subjects):
        data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
        labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
        if k != test_subject_id:
            X_s = data_subjects[k]
            y_s = labels_subjects[k]
            if args.align:
                X_s = EA(X_s)
            l_id = [i for i in range(y_s.shape[0]) if y_s[i] == 0]
            r_id = [i for i in range(y_s.shape[0]) if y_s[i] == 1]
            Xs0 = X_s[l_id]
            ys0 = y_s[l_id]
            Xs1 = X_s[r_id]
            ys1 = y_s[r_id]
            src_x.append(X_s)
            src_y.append(y_s)
            src_x_0.append(Xs0)
            src_y_0.append(ys0)
            src_x_1.append(Xs1)
            src_y_1.append(ys1)
            # print(len(src_x_0), Xs0.shape, len(src_x_1), Xs1.shape)
        else:
            nsamples = 1
            X_t = data_subjects.pop(test_subject_id)
            y_t = labels_subjects.pop(test_subject_id)
            if args.align:
                X_t = EA(X_t)
            l_index = [i for i in range(y_t.shape[0]) if y_t[i] == 0]
            r_index = [i for i in range(y_t.shape[0]) if y_t[i] == 1]
            X_la = X_t[l_index, :, :]
            X_ra = X_t[r_index, :, :]
            y_la = y_t[l_index]
            y_ra = y_t[r_index]
            # target train data
            X_lat = X_la[:nsamples, :, :]
            X_rat = X_ra[:nsamples, :, :]
            y_lat = y_la[:nsamples]
            y_rat = y_ra[:nsamples]
            # target test data
            X_lae = X_la[nsamples:, :, :]
            X_rae = X_ra[nsamples:, :, :]
            y_lae = y_la[nsamples:]
            y_rae = y_ra[nsamples:]
            X_tar_e = np.concatenate((X_lae, X_rae), axis=0)
            y_tar_e = np.concatenate((y_lae, y_rae), axis=0)
    if args.DWT_aug:
        src_aug_x = []
        src_aug_y = []
        wavename = 'db5'  # TODO db5
        TcA0, TcD0 = pywt.dwt(X_lat.squeeze(), wavename)
        TcA1, TcD1 = pywt.dwt(X_rat.squeeze(), wavename)
        for m in range(len(src_x_0)):
            Xs0 = src_x_0[m]
            for k in range(Xs0.shape[0]):
                Xss0 = Xs0[k, :, :]
                ScA0, ScD0 = pywt.dwt(Xss0, wavename)  # TODO dwt
                Xs_aug0 = pywt.idwt(ScA0, TcD0, 'db5', 'smooth')
                Xt_aug0 = pywt.idwt(TcA0, ScD0, 'db5', 'smooth')
                if args.align:
                    Xs_aug0 = EA(Xs_aug0)
                    Xt_aug0 = EA(Xt_aug0)
                src_aug_x.append(Xs_aug0[:, :Xss0.shape[-1]])
                src_aug_x.append(Xt_aug0[:, :Xss0.shape[-1]])
                src_aug_y.append(y_lat)
                src_aug_y.append(y_lat)
        for m in range(len(src_x_1)):
            Xs1 = src_x_1[m]
            for k in range(Xs1.shape[0]):
                Xss1 = Xs1[k]
                ScA1, ScD1 = pywt.dwt(Xss1, wavename)  # TODO dwt
                Xs_aug1 = pywt.idwt(ScA1, TcD1, 'db5', 'smooth')
                Xt_aug1 = pywt.idwt(TcA1, ScD1, 'db5', 'smooth')
                if args.align:
                    Xs_aug1 = EA(Xs_aug1)
                    Xt_aug1 = EA(Xt_aug1)
                src_aug_x.append(Xs_aug1[:, :Xss1.shape[-1]])
                src_aug_x.append(Xt_aug1[:, :Xss1.shape[-1]])
                src_aug_y.append(y_rat)
                src_aug_y.append(y_rat)
        src_aug_x = np.array(src_aug_x)
        src_aug_y = np.array(src_aug_y).squeeze()
        src_x = np.concatenate(src_x)
        src_y = np.concatenate(src_y)
        src_x = np.concatenate((src_x, src_aug_x))
        src_y = np.concatenate((src_y, src_aug_y))
        return src_x, src_y, X_t, y_t
    else:
        src_x = np.concatenate(src_x)
        src_y = np.concatenate(src_y)
        return src_x, src_y, X_t, y_t

def traintest_split_cross_subject_trialdiffnum(dataset, X, y, num_subjects, test_subject_id, trials_arr):
    if trials_arr:
        accum_arr = []
        for t in range(len(trials_arr)):
            accum_arr.append(np.sum([trials_arr[:(t + 1)]]))
        print(accum_arr)
        data_subjects = np.split(X, indices_or_sections=accum_arr, axis=0)
        labels_subjects = np.split(y, indices_or_sections=accum_arr, axis=0)
    else:
        data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
        labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def traintest_split_within_subject_Nsamples_diffnum(args, dataset, X, y, num_subjects, test_subject_id, nsamples, trials_arr):
    if 'Zhou2016' in dataset or 'HighGamma' in dataset or 'Weibo2014' in dataset:
        accum_arr = []
        for t in range(len(trials_arr)):
            accum_arr.append(np.sum([trials_arr[:(t + 1)]]))
        data_subjects = np.split(X, indices_or_sections=accum_arr, axis=0)
        labels_subjects = np.split(y, indices_or_sections=accum_arr, axis=0)
    else:
        data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
        labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    X_t = data_subjects.pop(test_subject_id)
    y_t = labels_subjects.pop(test_subject_id)
    l_index = [i for i in range(y_t.shape[0]) if y_t[i] == 0]
    r_index = [i for i in range(y_t.shape[0]) if y_t[i] == 1]
    X_la = X_t[l_index, :, :]
    X_ra = X_t[r_index, :, :]
    y_la = y_t[l_index]
    y_ra = y_t[r_index]
    X_lat = X_la[:nsamples, :, :]
    X_rat = X_ra[:nsamples, :, :]
    y_lat = y_la[:nsamples]
    y_rat = y_ra[:nsamples]
    X_lae = X_la[nsamples:, :, :]
    X_rae = X_ra[nsamples:, :, :]
    y_lae = y_la[nsamples:]
    y_rae = y_ra[nsamples:]
    X_tar_t = np.concatenate((X_lat, X_rat), axis=0)
    y_tar_t = np.concatenate((y_lat, y_rat), axis=0)
    X_tar_e = np.concatenate((X_lae, X_rae), axis=0)
    y_tar_e = np.concatenate((y_lae, y_rae), axis=0)
    print(X_tar_t.shape[0], X_tar_e.shape[0])
    if args.align:
        X_tar_t = EA(X_tar_t)
        X_tar_e = EA(X_tar_e)
    return X_tar_t, y_tar_t, X_tar_e, y_tar_e

def traintest_split_within_subject_Nsamples_samenum(args, X, y, num_subjects, test_subject_id, nsamples):
    data_subjects, labels_subjects = split_by_subject(args.data_name, X, y, num_subjects)
    X_t = data_subjects.pop(test_subject_id)
    y_t = labels_subjects.pop(test_subject_id)
    if args.data_name != 'BNCI2014001-4':
        l_index = [i for i in range(y_t.shape[0]) if y_t[i] == 0]
        r_index = [i for i in range(y_t.shape[0]) if y_t[i] == 1]
        X_la = X_t[l_index, :, :]
        X_ra = X_t[r_index, :, :]
        y_la = y_t[l_index]
        y_ra = y_t[r_index]
        X_lat = X_la[:nsamples, :, :]
        X_rat = X_ra[:nsamples, :, :]
        y_lat = y_la[:nsamples]
        y_rat = y_ra[:nsamples]
        X_lae = X_la[nsamples:, :, :]
        X_rae = X_ra[nsamples:, :, :]
        y_lae = y_la[nsamples:]
        y_rae = y_ra[nsamples:]
        X_tar_t = np.concatenate((X_lat, X_rat), axis=0)
        y_tar_t = np.concatenate((y_lat, y_rat), axis=0)
        X_tar_e = np.concatenate((X_lae, X_rae), axis=0)
        y_tar_e = np.concatenate((y_lae, y_rae), axis=0)
    else:
        l_index = [i for i in range(y_t.shape[0]) if y_t[i] == 0]
        r_index = [i for i in range(y_t.shape[0]) if y_t[i] == 1]
        f_index = [i for i in range(y_t.shape[0]) if y_t[i] == 2]
        t_index = [i for i in range(y_t.shape[0]) if y_t[i] == 3]
        X_l = X_t[l_index, :, :]
        X_r = X_t[r_index, :, :]
        X_f = X_t[f_index, :, :]
        X_t = X_t[t_index, :, :]
        y_l = y_t[l_index]
        y_r = y_t[r_index]
        y_f = y_t[f_index]
        y_t = y_t[t_index]
        X_lat = X_l[:nsamples, :, :]
        X_rat = X_r[:nsamples, :, :]
        X_fat = X_f[:nsamples, :, :]
        X_tat = X_t[:nsamples, :, :]
        y_lat = y_l[:nsamples]
        y_rat = y_r[:nsamples]
        y_fat = y_f[:nsamples]
        y_tat = y_t[:nsamples]
        X_lae = X_l[nsamples:, :, :]
        X_rae = X_r[nsamples:, :, :]
        X_fae = X_f[nsamples:, :, :]
        X_tae = X_t[nsamples:, :, :]
        y_lae = y_l[nsamples:]
        y_rae = y_r[nsamples:]
        y_fae = y_f[nsamples:]
        y_tae = y_t[nsamples:]
        X_tar_t = np.concatenate((X_lat, X_rat, X_fat, X_tat), axis=0)
        y_tar_t = np.concatenate((y_lat, y_rat, y_fat, y_tat), axis=0)
        X_tar_e = np.concatenate((X_lae, X_rae, X_fae, X_tae), axis=0)
        y_tar_e = np.concatenate((y_lae, y_rae, y_fae, y_tae), axis=0)
    print(X_tar_t.shape[0], X_tar_e.shape[0])
    if args.align:
        X_tar_t = EA(X_tar_t)
        X_tar_e = EA(X_tar_e)
    return X_tar_t, y_tar_t, X_tar_e, y_tar_e


def traintest_split_domain_classifier(dataset, X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    data_subjects.pop(test_subject_id)
    labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    for i in range(num_subjects - 1):
        labels_subjects[i] = np.ones((int(len(labels_subjects[i]))),) * i
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training:', train_x.shape, train_y.shape)
    return train_x, train_y, None, None


def traintest_split_domain_classifier_pretest(dataset, X, y, num_subjects, ratio):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    train_x_all = []
    train_y_all = []
    test_x_all = []
    test_y_all = []
    for i in range(num_subjects):
        data = data_subjects[i]
        random.shuffle(data)
        train_x_all.append(data[:int(len(data) * ratio)])
        train_y_all.append(np.ones((int(len(data) * ratio)),) * i)
        test_x_all.append(data[int(len(data) * ratio):])
        test_y_all.append(np.ones((int(len(data) * (1 - ratio))),) * i)
    train_x = np.concatenate(train_x_all, axis=0)
    train_y = np.concatenate(train_y_all, axis=0)
    test_x = np.concatenate(test_x_all, axis=0)
    test_y = np.concatenate(test_y_all, axis=0)
    print('Training/Test split:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y


def domain_split_multisource(dataset, X, y, num_subjects):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    print(num_subjects, 'Subjects of', data_subjects[0].shape, labels_subjects[0].shape)
    return data_subjects, labels_subjects
