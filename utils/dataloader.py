# -*- coding: utf-8 -*-
# @Time    : 2023/7/11
# @Author  : Siyang Li
# @File    : dataloader.py
import numpy as np
import torch
import mne
from sklearn import preprocessing
from sklearn.model_selection import KFold
from utils.data_utils import traintest_split_cross_subject, traintest_split_domain_classifier, domain_split_multisource, traintest_split_domain_classifier_pretest, traintest_split_cross_subject_trialdiffnum

from utils.data_utils import traintest_split_cross_subject_dwt

from utils.data_utils import traintest_split_cross_subject_dwt_class

from utils.data_utils import traintest_split_within_subject_Nsamples_diffnum, traintest_split_within_subject_Nsamples_samenum

from utils.data_utils import split_by_subject, subject_trial_counts, upeegst_dir


def data_process(args, dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if 'BNCI2014001' in dataset:
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    elif dataset == 'MI1-7':
        X = np.load('./data/' + 'MI1' + '/X-7.npy')
        y = np.load('./data/' + 'MI1' + '/labels-7.npy')
    elif dataset == 'MI1':
        X = np.load('./data/' + 'MI1' + '/X-7.npy')
        y = np.load('./data/' + 'MI1' + '/labels-7.npy')
    elif dataset.startswith('UP-EEGST-'):
        # One folder per session day, picked by the name suffix: UP-EEGST-ses07
        # is ./data/UP-EEGST/ses-07/ and so on (data/UP-EEGST/split_by_session.py).
        # Each folder is self-contained in the 004 format, so there is no
        # slicing to do here.
        root = upeegst_dir(dataset)
        X = np.load(root + '/X.npy')
        y = np.load(root + '/labels.npy')
    elif dataset.startswith('LU-EEGMI-'):
        # One folder per session, picked by the name suffix: LU-EEGMI-pre is
        # ./data/LU-EEGMI/pre/ and so on.  Each folder is self-contained in the
        # 004 format, so there is no slicing to do here.
        ses = dataset[len('LU-EEGMI-'):]
        X = np.load('./data/LU-EEGMI/' + ses + '/X.npy')
        y = np.load('./data/LU-EEGMI/' + ses + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    # print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22
        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:  # ['feet' 'left_hand' 'right_hand' 'tongue']
                indices.append(i)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 'left_hand')] = 0
        y[np.where(y == 'right_hand')] = 1
        y = y.astype(int)
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22
        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'Zhou2016':
        paradigm = 'MI'
        num_subjects = 4
        sample_rate = 250
        ch_num = 14
        presum_trials_arr = np.array([[179, 150, 150],
                                      [150, 135, 150],
                                      [150, 151, 150],
                                      [135, 150, 150]])

        # only use session 1
        indices = []
        trials_arr = []
        for i in range(num_subjects):
            inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
            # only use two classes [right_hand, feet]
            cnt = 0
            for j in inds:
                if y[j] in ['left_hand', 'right_hand']:  # TODO left_hand, right_hand, feet
                    indices.append(j)
                    cnt += 1
            trials_arr.append(cnt)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 'left_hand')] = 0
        y[np.where(y == 'right_hand')] = 1
        y = y.astype(int)
        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=3)  # TODO
    elif dataset == 'Zhou2016_3':
        paradigm = 'MI'
        num_subjects = 4
        sample_rate = 250
        ch_num = 14

        presum_trials_arr = np.array([[179, 150, 150],
                                      [150, 135, 150],
                                      [150, 151, 150],
                                      [135, 150, 150]])

        # only use session 1
        indices = []
        trials_arr = []
        for i in range(num_subjects):
            inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
            # only use two classes [right_hand, feet]
            cnt = 0
            for j in inds:
                if y[j] in ['left_hand', 'right_hand', 'feet']:  # TODO left_hand, right_hand, feet
                    indices.append(j)
                    cnt += 1
            trials_arr.append(cnt)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 'left_hand')] = 0
        y[np.where(y == 'right_hand')] = 1
        y[np.where(y == 'feet')] = 2
        y = y.astype(int)
        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=3)  # TODO
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=10)  # TODO
        if args.backbone == 'shallow':
            print('downsample...')
            X = mne.filter.resample(X, down=4)  # TODO
    elif dataset == 'BNCI2014004':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 3

        trials_arr = np.array([[120, 120, 160, 160, 160],  # 720
                               [120, 120, 160, 120, 160],  # 680
                               [120, 120, 160, 160, 160],  # 720
                               [120, 140, 160, 160, 160],  # 740
                               [120, 140, 160, 160, 160],  # 740
                               [120, 120, 160, 160, 160],  # 720
                               [120, 120, 160, 160, 160],  # 720
                               [160, 120, 160, 160, 160],  # 760
                               [120, 120, 160, 160, 160]])  # 720

        # only use session 1 first 120 trials, remove other sessions TODO
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(120) + np.sum(trials_arr[:i, :]))  # TODO 120
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'EEG-ST':
        paradigm = 'MI'
        # 9 of the 50 recordings.  Four (sub-05/13/15/49) never make it past
        # build_eegst.py -- three hold another subject's recording and one is
        # artifact in all 40 blocks.  Six more scored under 60% on all three CO
        # seeds, eight more had one seed standing 15+ points above the other
        # two, ten more had kappa at or below zero on at least one of five
        # seeds, and seven more had kappa below 0.3 on the single seed-1 run of
        # the remaining 16 (the earlier four single-seed cull rounds were undone
        # and are kept only as history); see data/EEG-ST/cull_report.txt.
        # Channels: the recorded 29 EEG rows were reduced in place to 3 on
        # 2026-10-03 -- Cz C3 C4, the C-row core of the earlier 6-channel
        # FCz FC3 FC4 / Cz C3 C4 set; the 6-, 15- and 22-channel steps of
        # the same day were superseded (list: data/EEG-ST/channels_3.txt,
        # 29-channel master copy: data/EEG-ST_29ch_backup/).
        num_subjects = 9
        sample_rate = 500
        ch_num = 3
        # X is already (9 subjects x <=40 blocks, 3, 4000), subject-major and
        # block-minor, with labels in {0, 1} straight from the .mat eeg.label,
        # so there is nothing to slice or relabel here.  Trials per subject are
        # ragged (24-40, AutoReject drops what it cannot repair); the split
        # helpers in utils/data_utils.py read meta.csv for the real boundaries.
    elif dataset.startswith('UP-EEGST'):
        paradigm = 'MI'
        # 'UP-EEGST' is the pooled folder -- all 14 subjects, all 278 sessions,
        # nothing culled -- and 'UP-EEGST-sesNN' is one session day out of it
        # (data/UP-EEGST/split_by_session.py).  Trials per subject are ragged
        # in both (1402 to 2820 pooled, 20 to 160 inside a session), so the
        # uniform np.split is wrong and the split helpers ask
        # subject_trial_counts instead -- utils/data_utils.py reads the real
        # boundaries off meta.csv, which is also where num_subjects comes from:
        # every session day up to ses-17 was attended by all 14 subjects, then
        # 13, 12, 9, and the last six by sub-03 alone.
        num_subjects = len(subject_trial_counts(dataset))
        sample_rate = 256
        ch_num = 11
    elif dataset.startswith('LU-EEGMI-'):
        paradigm = 'MI'
        # gait MI vs idle-state, 22 channels (core motor-imagery montage,
        # reduced in place from the recorded 40 on 2026-10-03; the later
        # 22 -> 15 step was reverted -- 22 performed best; list:
        # data/LU-EEGMI/channels_22.txt), 5 s window at 250 Hz, one folder
        # per paradigm (data/LU-EEGMI/preprocessing_info.txt).  Trials per
        # subject are ragged inside every folder -- 79 to 100 in pre, 20 to 100
        # in follow, 40 in MI-IES for sub-03/08 -- since the subjects ran
        # different numbers of runs, so, as with EEG-ST and UP-EEGST, the split
        # helpers ask subject_trial_counts() rather than dividing the row count
        # evenly.  The headcount is read back from meta.csv for the same reason
        # it is for UP-EEGST: it is 23 for pre/post/follow but 16 for the two
        # stim paradigms, and hard-coding either one would misdescribe the other.
        num_subjects = len(subject_trial_counts(dataset))
        sample_rate = 250
        ch_num = 22
    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=10)  # TODO
        if args.backbone == 'shallow':
            print('downsample...')
            X = mne.filter.resample(X, down=4)  # TODO
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'MI1-7':
        paradigm = 'MI'
        num_subjects = 7
        sample_rate = 1000
        ch_num = 59
        print('MI downsampled')
        X = mne.filter.resample(X, down=4)
        sample_rate = int(sample_rate // 4)

        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=3)  # TODO
            sample_rate = int(sample_rate // 3)
    elif dataset == 'MI1':
        paradigm = 'MI'
        num_subjects = 5
        sample_rate = 1000
        ch_num = 59
        print('MI downsampled')
        X = mne.filter.resample(X, down=4)
        sample_rate = int(sample_rate // 4)
        X = np.concatenate([X[200:1000, :, :], X[1200:1400, :, :]], axis=0)  # S0 and S5 is left hand/right foot
        y = np.concatenate([y[200:1000], y[1200:1400]], axis=0)  # S0 and S5 is left hand/right foot
        if args.backbone == 'deep':
            print('downsample...')
            X = mne.filter.resample(X, down=3)  # TODO
            sample_rate = int(sample_rate // 3)
    elif 'BNCI2014008' in dataset:
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8
        class_num = 2

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    if dataset == 'Zhou2016':
        return X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr
    else:
        return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_process_secondsession(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session E, remove session T
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i) + 288) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session test, remove session train
        indices = []
        for i in range(num_subjects):
            #indices.append(np.arange(100) + (160 * i))
            indices.append(np.arange(60) + (160 * i) + 100) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            # use second sessions
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7) + 200)
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7) + 200)
            else:
                indices.append(np.arange(200) + (400 * i) + 200)

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session E, remove session T
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i) + 288)
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def read_mi_combine_tar(args):

    if args.data_name == 'Zhou2016':
        X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args, args.data)
        src_data, src_label, tar_data, tar_label = traintest_split_cross_subject_trialdiffnum(args.data, X, y, num_subjects, args.idt, trials_arr)
        args.trials_arr = trials_arr
    else:
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args, args.data)
        src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)
        # EEG-ST keeps a different number of trials per subject, so the source
        # set has to be aligned subject by subject.  Handing data_loader the
        # per-subject counts is what selects data_alignment_uneven over
        # data_alignment, which would otherwise divide the concatenation by
        # args.N - 1 and cut every subject in the wrong place (utils/utils.py).
        counts = subject_trial_counts(args.data)
        args.trials_arr = list(counts) if counts is not None else False

    return src_data, src_label, tar_data, tar_label

def read_mi_within_tar(args):

    if args.data_name == 'Zhou2016':
        X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args, args.data)
        src_data, src_label, tar_data, tar_label = traintest_split_within_subject_Nsamples_diffnum(args, args.data, X, y, num_subjects, args.idt, args.nsamples, trials_arr)
        args.trials_arr = trials_arr
    else:
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args, args.data)
        src_data, src_label, tar_data, tar_label = traintest_split_within_subject_Nsamples_samenum(args, X, y, num_subjects, args.idt, args.nsamples)
        # Same reason as read_mi_combine_tar above: on a ragged dataset the
        # source concatenation has to be aligned subject by subject, and these
        # per-subject counts are what selects data_alignment_uneven over
        # data_alignment (utils/utils.py).
        counts = subject_trial_counts(args.data)
        args.trials_arr = list(counts) if counts is not None else False

    return src_data, src_label, tar_data, tar_label

def read_mi_within_tar_CV(args):
    if args.data_name == 'Zhou2016':
        X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args, args.data)
        accum_arr = []
        for t in range(len(trials_arr)):
            accum_arr.append(np.sum([trials_arr[:(t + 1)]]))
        data_subjects = np.split(X, indices_or_sections=accum_arr, axis=0)
        labels_subjects = np.split(y, indices_or_sections=accum_arr, axis=0)
        X_t = data_subjects.pop(args.idt)
        y_t = labels_subjects.pop(args.idt)
        args.trials_arr = trials_arr
    else:
        X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args, args.data)
        data_subjects, labels_subjects = split_by_subject(args.data, X, y, num_subjects)
        X_t = data_subjects.pop(args.idt)
        y_t = labels_subjects.pop(args.idt)
        args.trials_arr = False
    return X_t, y_t

def read_mi_combine_tar_dwt(args):
    # if args.data_name == 'Zhou2016':
    #     X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args, args.data)
    #     src_data, src_label, tar_data, tar_label = traintest_split_cross_subject_trialdiffnum(args.data, X, y, num_subjects, args.idt, trials_arr)
    #     args.trials_arr = trials_arr
    # else:
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args, args.data)
    src_data, src_label, tar_data, tar_label = traintest_split_cross_subject_dwt_class(args, X, y, num_subjects, args.idt)
    args.trials_arr = False

    return src_data, src_label, tar_data, tar_label


def read_mi_combine_tar_secondsession(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process_secondsession(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def read_mi_combine_domain(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def read_mi_combine_domain_split(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier_pretest(args.data, X, y, num_subjects, args.ratio)

    return src_data, src_label, tar_data, tar_label


def read_mi_multi_source(args):
    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process(args.data)

    data_subjects, labels_subjects = domain_split_multisource(args.data, X, y, num_subjects)

    return data_subjects, labels_subjects


def data_normalize(fea_de, norm_type):
    if norm_type == 'zscore':
        zscore = preprocessing.StandardScaler()
        fea_de = zscore.fit_transform(fea_de)

    return fea_de
