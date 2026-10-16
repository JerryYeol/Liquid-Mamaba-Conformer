'''
=================================================
coding:utf-8
@Time:      2025/4/28 19:36
@File:      DBConformer_CV.py
@Author:    Ziwei Wang
@Function:  Cross-Validation (CV) scenario
=================================================
'''
import math
import copy
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from utils.network import backbone_net, backbone_net_deep, backbone_net_shallow, backbone_net_ifnet, \
    backbone_net_fbcnet, backbone_net_adfcnn, backbone_net_conformer, backbone_net_dbconformer, backbone_net_fbmsnet, \
    backbone_net_CTNet, backbone_net_eegnex, backbone_net_wave, backbone_net_mscformer, backbone_net_msvtnet, \
    backbone_net_slimseiz, backbone_net_scnn_v2, backbone_net_tmsanet, \
    measure_feature_dim, smallest_divisor_at_least
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_within_tar_CV
from utils.data_utils import subject_trial_counts
from utils.utils import fix_random_seed, cal_acc_comb, cal_acc_comb_full, data_loader_within
from utils.alg_utils import EA
from utils.metrics import (compute_metrics, metrics_to_row, format_metrics, run_paths,
                           append_csv, update_json, save_probabilities, summarise_csv)
from sklearn.model_selection import KFold
import gc
import sys
import warnings
warnings.filterwarnings('ignore')

# number of cross-validation folds per subject
KFOLD_SPLITS = 5


def train_target(args):
    X, y = read_mi_within_tar_CV(args)
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=False)  # Important, cannot shuffle for MI classification
    acc_cv = []
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_src, X_tar = X[train_index], X[test_index]
        y_src, y_tar = y[train_index], y[test_index]
        if args.align:
            X_src = EA(X_src)
            X_tar = EA(X_tar)
        print("训练集大小:", X_src.shape, "测试集大小:", X_tar.shape)
        dset_loaders = data_loader_within(X_src, y_src, X_tar, y_tar, args)
        # network selection
        if args.backbone == 'EEGNet':
            netF, netC = backbone_net(args, return_type='xy')
        elif args.backbone == 'deep':
            netF, netC = backbone_net_deep(args, return_type='xy')
        elif args.backbone == 'shallow':
            netF, netC = backbone_net_shallow(args, return_type='xy')
        elif args.backbone == 'IFNet':
            netF, netC = backbone_net_ifnet(args, return_type='xy')
        elif args.backbone == 'FBCNet':
            netF = backbone_net_fbcnet(args, return_type='xy')
        elif args.backbone == 'ADFCNN':
            netF, netC = backbone_net_adfcnn(args, return_type='xy')
        elif args.backbone == 'Conformer':
            netF = backbone_net_conformer(args)
        elif args.backbone == 'DBConformer':
            netF = backbone_net_dbconformer(args)
        elif args.backbone == 'FBMSNet':
            netF, netC = backbone_net_fbmsnet(args, return_type='xy')
        elif args.backbone == 'CTNet':
            netF, netC = backbone_net_CTNet(args, return_type='xy')
        elif args.backbone == 'EEGNeX':
            netF, netC = backbone_net_eegnex(args, return_type='xy')
        elif args.backbone == 'EEGWaveNet':
            netF = backbone_net_wave(args)
        elif args.backbone == 'MSCFormer':
            netF = backbone_net_mscformer(args, return_type='xy')
        elif args.backbone == 'MSVTNet':
            netF = backbone_net_msvtnet(args, return_type='xy')
        elif args.backbone == 'SlimSeiz':
            netF = backbone_net_slimseiz(args, return_type='xy')
        elif args.backbone == 'ShallowConvNet_V2':
            netF, netC = backbone_net_scnn_v2(args, return_type='xy')
        elif args.backbone == 'TMSANet':
            netF = backbone_net_tmsanet(args, return_type='xy')
        # base_network is always built, then moved to GPU only when running there.
        # The eight below carry their own classifier and their forward already
        # returns (features, logits), so there is no head to stack on top of them.
        if args.backbone in ('FBCNet', 'Conformer', 'DBConformer', 'EEGWaveNet',
                             'MSCFormer', 'MSVTNet', 'SlimSeiz', 'TMSANet'):
            base_network = netF
        else:
            base_network = nn.Sequential(netF, netC)
        if args.data_env != 'local':
            base_network = base_network.cuda()
        optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
        if args.backbone not in ('FBCNet', 'Conformer', 'DBConformer', 'EEGWaveNet',
                                 'MSCFormer', 'MSVTNet', 'SlimSeiz', 'TMSANet'):
            optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)
        if args.class_num == 2:
            class_weight = torch.tensor([1., args.weight], dtype=torch.float32)  # class imbalance
            if args.data_env != 'local':
                class_weight = class_weight.cuda()
            criterion = nn.CrossEntropyLoss(weight=class_weight)
        else:
            criterion = nn.CrossEntropyLoss()
        max_iter = args.max_epoch * len(dset_loaders["source"])
        interval_iter = max_iter // args.max_epoch
        args.max_iter = max_iter
        iter_num = 0
        base_network.train()

        # ---- per-epoch bookkeeping -----------------------------------------
        # DBConformer (ours) is scored on its best epoch: the highest target ACC
        # reached during training, which is what we report and the weights we
        # keep.  Every comparison backbone reports its *last* epoch instead --
        # the network the training loop leaves standing, with no epoch
        # selection at all -- so nothing per-epoch has to be kept for those;
        # see the selection after the loop.
        best_acc, best_epoch, best_state = -float('inf'), -1, None
        while iter_num < max_iter:
            try:
                inputs_source, labels_source = next(iter_source)
            except:
                iter_source = iter(dset_loaders["source"])
                inputs_source, labels_source = next(iter_source)
            if inputs_source.size(0) == 1:
                continue
            iter_num += 1
            # Has to stay word-for-word identical to the reshape in
            # utils/utils.py:_cal_acc_comb_core, which is the eval path's only
            # copy of this decision.  EEGNet / deep / shallow are absent on
            # purpose: data_loader already gave those three a channel axis.
            # EEGNeX, EEGWaveNet, FBCNet, IFNet and TMSANet take (B, C, T) as is.
            if args.backbone == 'FBMSNet':
                # dim 1 is the band axis for FBMSNet, not a singleton
                inputs_source = inputs_source.reshape(inputs_source.shape[0], args.in_chans,
                                                      args.chn // args.in_chans, -1)
            elif args.backbone in ('ADFCNN', 'CTNet', 'Conformer', 'DBConformer',
                                   'MSCFormer', 'MSVTNet', 'ShallowConvNet_V2', 'SlimSeiz'):
                inputs_source = inputs_source.unsqueeze_(3)
                inputs_source = inputs_source.permute(0, 3, 1, 2)
            features_source, outputs_source = base_network(inputs_source)
            classifier_loss = criterion(outputs_source, labels_source)
            optimizer_f.zero_grad()
            if args.backbone not in ('FBCNet', 'Conformer', 'DBConformer', 'EEGWaveNet',
                                     'MSCFormer', 'MSVTNet', 'SlimSeiz', 'TMSANet'):
                optimizer_c.zero_grad()
            classifier_loss.backward()
            optimizer_f.step()
            if args.backbone not in ('FBCNet', 'Conformer', 'DBConformer', 'EEGWaveNet',
                                     'MSCFormer', 'MSVTNet', 'SlimSeiz', 'TMSANet'):
                optimizer_c.step()
            if iter_num % interval_iter == 0 or iter_num == max_iter:
                base_network.eval()
                epoch_idx = int(iter_num // len(dset_loaders["source"]))
                acc_t_te, _ = cal_acc_comb(dset_loaders["target-online"], base_network, args=args)  # TODO target-online
                # strictly greater, so on a tie the earlier epoch (less overfit) wins
                is_best = acc_t_te > best_acc
                if is_best:
                    best_acc, best_epoch = acc_t_te, epoch_idx
                    best_state = copy.deepcopy(base_network.state_dict())
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%{}'.format(args.task_str,
                                                                       epoch_idx,
                                                                       int(max_iter // len(dset_loaders["source"])),
                                                                       acc_t_te,
                                                                       '   <-- best' if is_best else '')
                args.log.record(log_str)
                base_network.train()

        # ---- which epoch is this fold's result ----------------------------
        # DBConformer (ours) reports its best epoch: the highest target ACC
        # reached during training.  Every comparison backbone reports its *last*
        # epoch instead -- no epoch selection at all, just the network as
        # training left it -- so there is nothing to restore for those.  Either
        # way the weights standing here are the ones that produced everything
        # below: the ACC, the metric set, the saved probabilities and the
        # checkpoint.
        if args.backbone == 'DBConformer':
            if best_state is not None:
                base_network.load_state_dict(best_state)
            result_epoch, result_kind = best_epoch, 'Best'
        else:
            # training stopped at the last epoch, so the live weights are the result
            result_epoch, result_kind = int(max_iter // len(dset_loaders["source"])), 'Last'
        base_network.eval()
        acc_final, prob, true = cal_acc_comb_full(dset_loaders["target-online"], base_network, args=args)
        metrics = compute_metrics(true.numpy(), prob.numpy(), n_classes=args.class_num)
        if result_kind == 'Best':
            print('Fold {}/{}  Best epoch {}/{} -> ACC = {:.2f}%   (last epoch {:.2f}%, +{:.2f})'.format(
                fold, kf.get_n_splits(), result_epoch,
                int(max_iter // len(dset_loaders["source"])),
                acc_final, acc_t_te, acc_final - acc_t_te))
        else:
            print('Fold {}/{}  Last epoch {}/{} -> ACC = {:.2f}%'.format(
                fold, kf.get_n_splits(), result_epoch,
                int(max_iter // len(dset_loaders["source"])), acc_final))
        print('  ' + format_metrics(metrics))
        args.log.record('  ' + format_metrics(metrics))

        # ---- persist: model, per-fold CSV row, JSON detail, raw probs -----
        print('saving model (epoch {})...'.format(result_epoch))
        # DBC_RUNS_DIR is the sensitivity-sweep hook: run_sensitivity.sh points
        # it at one folder per hyperparameter configuration so a sweep never
        # overwrites the main runs/.  Unset, this is still ./runs/.
        if not os.path.exists(args.runs_dir + str(args.data_name) + '/'):
            os.makedirs(args.runs_dir + str(args.data_name) + '/')
        ckpt = (args.runs_dir + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt)
                + '_seed' + str(args.SEED) + '_fold' + str(fold)
                + ('' if args.align else '_noEA') + '.ckpt')
        # base_network holds the result epoch's weights by now, whichever mode ran
        torch.save(base_network.state_dict(), ckpt)

        paths = getattr(args, 'metric_paths', None) or run_paths(args)
        key = 'seed{}_S{}_fold{}'.format(args.SEED, args.idt, fold)
        meta = dict(data_name=args.data_name, method=args.method, backbone=args.backbone,
                    paradigm=args.paradigm, seed=args.SEED, idt=args.idt, fold=fold,
                    task_str=args.task_str, best_epoch=best_epoch,
                    last_epoch_acc=acc_t_te, acc_best=acc_final)
        append_csv(paths['csv'], metrics_to_row(meta, metrics))
        update_json(paths['json'], key, dict(meta, **metrics, ckpt=ckpt))
        save_probabilities(paths['prob_dir'], key, prob.numpy(), true.numpy())

        acc_cv.append(acc_final)
        gc.collect()
        if args.data_env != 'local':
            torch.cuda.empty_cache()
    print('  metrics -> {}'.format(paths['csv']))
    print(acc_cv)
    return np.mean(acc_cv)


if __name__ == '__main__':
    cpu_num = 8
    torch.set_num_threads(cpu_num)
    # One folder per session day of UP-EEGST: data/UP-EEGST/ses-01 .. ses-26,
    # addressed as 'UP-EEGST-ses01' .. 'UP-EEGST-ses26' -- each holds every
    # subject who attended that day, 14 down to sub-03 on its own.  Plain
    # 'UP-EEGST' would be the pooled 278-session folder instead.  Other
    # datasets: 'BNCI2014001', 'BNCI2014004', 'Zhou2016', 'MI1-7',
    # 'BNCI2014002', 'EEG-ST', 'LU-EEGMI-pre' ...
    # DBC_DATA is how run_baselines.sh picks the dataset per process; with
    # the variable unset the literal here is what runs.
    data_name_list = [os.environ.get('DBC_DATA', 'LU-EEGMI-follow')]# 'UP-EEGST-ses{:02d}'.format(k) for k in range(1, 27)
    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])
    for data_name in data_name_list:
        weight = 1
        # Every branch below sets trial_num; the guard after the table reads it.
        trial_num = None
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 22, 2, 1001, 250, 144
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 14, 15, 2, 2561, 512, 100
        if data_name == 'BNCI2014004': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 3, 2, 1126, 250, 120
        if data_name == 'EEG-ST': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 3, 2, 4000, 500, 40
        # UP-EEGST rows are whole 4 s MI trials at 256 Hz, and the folders are
        # cut by session day: data/UP-EEGST/ses-01 .. ses-26, addressed as
        # 'UP-EEGST-ses01' .. 'UP-EEGST-ses26' (data/UP-EEGST/split_by_session.py),
        # plus the pooled data/UP-EEGST/ itself, addressed as plain 'UP-EEGST'.
        # Nothing is culled -- all 14 subjects, 23382 trials, including the
        # 1.66% of samples sitting on the +-500 uV clip ceiling an earlier
        # build dropped sub-08 for.  Trials per subject are ragged inside every
        # folder (1402-2820 pooled, 20-160 in a session) and so is the subject
        # count: ses-01..ses-17 hold all 14, ses-18 holds 13, ses-19 holds 12,
        # ses-20 holds 9, and ses-21..ses-26 are sub-03 alone.  Both numbers
        # therefore come off the folder's meta.csv instead of being written
        # down here, as does every caller that needs a real trial count
        # (subject_trial_counts / subject_minority_count, utils/data_utils.py);
        # np.split would cut this dataset in the wrong place.  time_sample_num
        # is the 4 s the paradigm's motor-imagery phase lasts
        # (task-motorimagery_events.json, and the authors' own
        # preprocess_derivatives MI_START_SECONDS=9 / MI_END_SECONDS=13), which
        # is what the other four datasets here already feed.
        if data_name.startswith('UP-EEGST'):
            paradigm, chn, class_num, time_sample_num, sample_rate = 'MI', 11, 2, 1024, 256
            _counts = subject_trial_counts(data_name)
            N, trial_num = len(_counts), min(_counts)
        # LU-EEGMI: gait MI vs idle-state, 22 channels, 5 s window at 250 Hz,
        # X.npy now holds the 22 core motor-imagery electrodes only -- the
        # recorded 40 channels were reduced in place on 2026-10-03 (a 22 -> 15
        # step tried the same day was reverted: 22 performed best; list:
        # data/LU-EEGMI/channels_22.txt, 40-channel copy:
        # data/LU-EEGMI_40ch_backup/, rebuild: restore_22_channels.py).
        # one folder per paradigm with the folder picked by the name suffix
        # (data/LU-EEGMI/{pre,post,follow,MI-SES,MI-IES}/).  Trials per subject
        # are ragged inside every folder -- 79-100 in pre, 20-100 in follow, 40
        # for sub-03/08 in MI-IES -- so trial_num below is only a summary and
        # callers that need a real count ask subject_trial_counts() /
        # subject_minority_count(), which read the boundaries off meta.csv
        # (utils/data_utils.py).  pre/post/follow hold the same 23 subjects
        # (those with all three visits, sub-03/06/08/11 never had the last
        # two); MI-SES/MI-IES hold the 16 who have both stim visits, a different
        # cohort, which is why N is not shared across the five.
        if data_name == 'LU-EEGMI-pre': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 23, 22, 2, 1250, 250, 80
        if data_name == 'LU-EEGMI-post': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 23, 22, 2, 1250, 250, 80
        if data_name == 'LU-EEGMI-follow': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 23, 22, 2, 1250, 250, 80
        if data_name == 'LU-EEGMI-MI-SES': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 16, 22, 2, 1250, 250, 80
        if data_name == 'LU-EEGMI-MI-IES': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 16, 22, 2, 1250, 250, 40
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 12, 13, 2, 2561, 512, 200
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 9, 22, 4, 1001, 250, 288
        if data_name == 'MI1-7': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 7, 59, 2, 750, 250, 200
        if data_name == 'MI1': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, dim_e, dim_p = 'MI', 5, 59, 2, 750, 250, 200, 184, 750
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, weight = 'ERP', 8, 8, 2, 256, 256, 4200, 64,
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, weight = 'ERP', 10, 8, 2, 206, 256, 2520, 64
        if data_name == 'Zhou2016': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 4, 14, 2, 1251, 250, -1
        if data_name == 'Zhou2016_3': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num = 'MI', 4, 14, 3, 1251, 250, -1
        # A name no branch above matched used to surface twenty lines later as
        # "NameError: name 'trial_num' is not defined"; say what is wrong instead.
        if trial_num is None:
            raise ValueError('unknown dataset: ' + data_name
                             + ' (no branch for it in the dataset table above)')
        F1, D, F2 = 4, 2, 8

        if 'BNCI2014008' in data_name:
            F1, D, F2 = 8, 4, 16
        args = argparse.Namespace(trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name,
                                  F1=F1, D=D, F2=F2, weight=weight)

        # DBC_BACKBONE is how run_baselines.sh sweeps the comparison
        # backbones; with the variable unset this line is what runs.
        args.backbone = os.environ.get('DBC_BACKBONE', 'DBConformer')  # DBConformer (Ours)
        args.method = args.backbone + '_' + data_name
        # DBConformer parameters.
        # DBC_GATE_FLAG / DBC_POSEMB_FLAG / DBC_CHN_ATTEN_FLAG / DBC_BRANCH /
        # DBC_CFC_FLAG are the ablation hooks that run_ablation.sh sets, one
        # module at a time; with the variables unset these lines are exactly the
        # values that ran before (gate off, PE on, channel attention on, both
        # branches, CfC on), so a normal run is unchanged.
        args.gate_flag = os.environ.get('DBC_GATE_FLAG', '0') == '1'
        args.posemb_flag = os.environ.get('DBC_POSEMB_FLAG', '1') == '1'
        args.chn_atten_flag = os.environ.get('DBC_CHN_ATTEN_FLAG', '1') == '1'
        args.branch = os.environ.get('DBC_BRANCH', 'all')  # [all, temporal, spatial]
        assert args.branch in ('all', 'temporal', 'spatial'), args.branch
        args.cfc_flag = os.environ.get('DBC_CFC_FLAG', '1') == '1'
        if args.backbone == 'DBConformer':
            args.emb_size = 40
            args.spa_dim = 16
            # DBC_DEPTH / DBC_PATCH_SIZE are the sensitivity-sweep hooks that
            # run_sensitivity.sh sets, one hyperparameter at a time; with the
            # variables unset these branches pick the same values as before.
            args.transformer_depth_tem = int(os.environ.get('DBC_DEPTH', 2))
            args.transformer_depth_chn = int(os.environ.get('DBC_DEPTH', 2))
            if data_name == 'BNCI2015001' or data_name == 'BNCI2014002':
                patch_default = 128
            else:
                patch_default = 125
            args.patch_size = int(os.environ.get('DBC_PATCH_SIZE', patch_default))
        # whether to use EA
        args.align = True
        args.dropoutRate = 0.25
        # learning rate
        args.lr = 0.001
        # batch size: DBC_BATCH_SIZE is the sensitivity-sweep hook; unset it is 32
        args.batch_size = int(os.environ.get('DBC_BATCH_SIZE', 32))
        # training epochs
        args.max_epoch = 100
        # GPU device id (optional): pass an index to pin a GPU, or 'cpu' to force CPU.
        # With no argument the first visible GPU is used whenever one is available,
        # so launching from an IDE (which passes no argv) no longer falls back to CPU.
        device_id = str(sys.argv[1]) if len(sys.argv) > 1 else '0'
        force_cpu = device_id.lower() == 'cpu'
        if not force_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        args.data_env = 'gpu' if (not force_cpu and torch.cuda.device_count() != 0) else 'local'
        print('data_env:', args.data_env, '| CUDA_VISIBLE_DEVICES:',
              os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'))
        # ---- per-backbone construction parameters -----------------------------
        # Everything the factories in utils/network.py read is set here, so
        # switching backbones is a one-line change on line 281.  The factories
        # only read these attributes; they never patch args themselves.
        args.dropout_rate = args.dropoutRate   # models read either spelling
        args.pooling_size = 52                 # MSCFormer: avg-pool width (its default)
        args.embed_dims = 16                   # IFNet: out_planes
        args.embed_dim = 19                    # TMSANet: embedding width (its default)
        args.nBands = smallest_divisor_at_least(args.chn, 2)   # FBCNet: has to divide C
        args.feature_deep_dim = 1
        if args.backbone == 'Conformer':
            # models/Conformer.py builds its head inside __init__, so this width
            # cannot be measured after the fact: 40 filters through conv (1,25)
            # valid (-> T-24) and AvgPool2d((1,75),(1,15)) (-> (T-99)//15+1).
            args.feature_deep_dim = 40 * ((args.time_sample_num - 99) // 15 + 1)
        elif args.backbone == 'FBCNet':
            # FBCNet folds the time axis into (strideFactor, T // strideFactor)
            # after dropping 2 samples (1 on BNCI2014004), so strideFactor has to
            # divide what is left.  It drops a channel, not a sample, on
            # BNCI2015001 / MI1-7 / MI1 -- that one leaves the time axis alone.
            trimmed = args.time_sample_num
            if 'BNCI2014001' in args.data_name or args.data_name == 'BNCI2014002' or args.data_name == 'BNCI2015001':
                trimmed = trimmed - 2
            elif 'BNCI2014004' in args.data_name:
                trimmed = trimmed - 1
            args.strideFactor = next((d for d in (3, 2, 5, 4, 6, 7, 8) if trimmed % d == 0), 1)
        elif args.backbone == 'IFNet':
            # The stem drops one time sample and reshapes to (T // patch_size,
            # patch_size), so the patch has to divide T - 1.
            args.patch_size = next((p for p in (125, 100, 128, 200, 250, 500)
                                    if args.time_sample_num % p == 1), args.time_sample_num - 1)
        elif args.backbone == 'FBMSNet':
            # There is no filter bank in this data, so the band axis is faked by
            # splitting the electrodes into in_chans groups (MixedConv2d takes
            # one branch per group, four at most).  feature_deep_dim is the
            # MixedConv2d width here, not a head width -- backbone_net_fbmsnet
            # derives the head width from it.
            args.in_chans = smallest_divisor_at_least(args.chn, 4)
            args.strideFactor = next((d for d in (3, 2, 5, 4, 6, 7, 8)
                                      if args.time_sample_num % d == 0), 1)
            args.feature_deep_dim = 36
        if args.backbone in ('EEGNet', 'deep', 'shallow', 'IFNet', 'ADFCNN', 'CTNet',
                             'EEGNeX', 'ShallowConvNet_V2'):
            # a separate if, not the last elif: IFNet has a branch above and still
            # needs a width.  Measured rather than derived -- see
            # utils/network.py:measure_feature_dim for why.
            args.feature_deep_dim = measure_feature_dim(
                args, device='cuda' if args.data_env == 'gpu' else 'cpu')
        total_acc = []
        # train_target scores DBConformer on its best epoch and the comparison
        # backbones on their last epoch; this only labels the lines below.
        epoch_kind = 'best' if args.backbone == 'DBConformer' else 'last'
        # 跑哪些随机种子：DBC_SEEDS="1" 就只跑 seed 1；不设时仍是五个全跑。
        # 和 DBC_BACKBONE / DBC_DATA 一样是环境变量钩子，手敲 python 的行为不变。
        for s in [int(x) for x in os.environ.get('DBC_SEEDS', '1 2 3 4 5').replace(',', ' ').split()]:
            args.SEED = s
            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True
            args.data = data_name
            print(args.data)
            print(args.method)
            print(args.SEED)
            print(args)
            args.local_dir = './data/' + str(data_name) + '/'
            # DBC_RESULT_DIR keeps a sweep's log + per-fold metrics out of the
            # main logs/ tree; unset, this is still ./logs/.
            args.result_dir = os.environ.get('DBC_RESULT_DIR', './logs/')
            args.runs_dir = os.environ.get('DBC_RUNS_DIR', './runs/')
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)
            # per-fold metrics for this run: <log name>_subjects.csv / _detail.json / _probs/
            args.metric_paths = run_paths(args)
            print('metrics ->', args.metric_paths['csv'])
            my_log.record('metrics -> ' + args.metric_paths['csv'])
            sub_acc_all = np.zeros(N)
            for idt in range(N):
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)
                args.log = my_log
                if args.data_name == 'Zhou2016':
                    sbj_num = [119, 100, 100, 90]
                    args.nsamples = math.ceil(sbj_num[idt] / 2 * 0.8)  # 80% training，20% testing
                elif args.data_name == 'BNCI2014001-4':
                    args.nsamples = math.ceil(args.trial_num / 4 * 0.8)
                else:
                    args.nsamples = math.ceil(args.trial_num / 2 * 0.8)
                sub_acc_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)
            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record('per-subject ACC ({} epoch, seed {}): '.format(epoch_kind, args.SEED) + acc_sub_str)
            args.log.record('mean ACC: ' + acc_mean_str)
            # ---- full metric summary over folds --------------------------------
            summary = summarise_csv(args.metric_paths['csv'])
            print('---- per-fold {}-epoch metrics (n={} subjects x {} folds) ----'.format(epoch_kind, N, KFOLD_SPLITS))
            args.log.record('---- per-fold {}-epoch metrics (n={} subjects x {} folds) ----'.format(epoch_kind, N, KFOLD_SPLITS))
            for k, (mu, sd) in summary.items():
                line = '{:20s} {:.3f} +- {:.3f}'.format(k, mu, sd)
                print('  ' + line)
                args.log.record('  ' + line)
        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)
        print(str(total_acc))
        args.log.record(str(total_acc))
        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)
        print(subject_mean)
        print(args.method)
        print(total_mean)
        print(total_std)
        args.log.record(str(subject_mean))
        args.log.record(str(total_mean))
        args.log.record(str(total_std))