from utils import GradCAM
import matplotlib.pyplot as plt
import torch
import numpy as np


def PlotHIST(args, Prediction_Loader, model, device):
    if args.Single:
        # model.eval()
        cam_ori = GradCAM(
            model=model,
            target_layer=model.OriginalBackbone[-1]
        )

        cam_cal = GradCAM(
            model=model,
            target_layer=model.CalibrationBackbone[-1]
        )
        ori_pos_feats = []
        ori_neg_feats = []

        cal_pos_feats = []
        cal_neg_feats = []

        MAX_SAMPLES = 3  # 最多五个样本
        count_pos = 0
        count_neg = 0

        for batch_idx, (recompressed, sample, labels) in enumerate(Prediction_Loader):
            recompressed = recompressed.to(device)
            sample = sample.to(device)
            labels = labels.long().to(device)

            with torch.no_grad():  # ⚠️ 这里不需要梯度
                _ = model(recompressed, sample)

            r = cam_ori.feat[0].detach().cpu().numpy().reshape(-1)  # (S, C)
            s = cam_cal.feat[0].detach().cpu().numpy().reshape(-1)

            if labels.item() == 1 and count_pos < MAX_SAMPLES:
                ori_pos_feats.append(r)
                cal_pos_feats.append(s)
                count_pos += 1
            elif labels.item() == 0 and count_neg < MAX_SAMPLES:
                ori_neg_feats.append(r)
                cal_neg_feats.append(s)
                count_neg += 1

            if (batch_idx % 1) == 0:
                print()
                print(f'[INFO] 测试到第{batch_idx}/{9600 / args.batch_size}个Batch')
            if batch_idx == 32:
                break

        ori_pos_feats = np.concatenate(ori_pos_feats)
        ori_neg_feats = np.concatenate(ori_neg_feats)

        cal_pos_feats = np.concatenate(cal_pos_feats)
        cal_neg_feats = np.concatenate(cal_neg_feats)

        plt.figure(figsize=(8, 8))
        plt.hist(
            cal_pos_feats,
            bins=100,
            alpha=0.6,
            density=False,
            label='Stego'
        )
        plt.hist(
            cal_neg_feats,
            bins=100,
            alpha=0.6,
            density=False,
            label='Cover'
        )

        plt.title("Calibration Backbone Feature Distribution")
        plt.xlabel("Feature")
        plt.ylabel("Frequency")
        plt.legend(frameon=False)
        plt.grid(True)
        plt.xlim(-2, 2)  # 你可以根据实际调
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.hist(
            ori_pos_feats,
            bins=100,
            alpha=0.6,
            density=False,
            label='Stego'
        )
        plt.hist(
            ori_neg_feats,
            bins=100,
            alpha=0.6,
            density=False,
            label='Cover'
        )

        plt.title("Original Backbone Feature Distribution")
        plt.xlabel("Feature")
        plt.ylabel("Frequency")
        plt.legend(frameon=False)
        plt.grid(True)
        plt.xlim(-2, 2)  # 你可以根据实际调
        plt.show()


    else:
        MAX_SAMPLES = 3  # 最多五个样本
        count_pos = 0
        count_neg = 0

        cam_ori = GradCAM(
            model=model,
            target_layer=model.OriginalBackbone[-1]
        )

        cam_cal = GradCAM(
            model=model,
            target_layer=model.CalibrationBackbone[-1]
        )

        ori_pos, ori_neg = [], []
        cal_pos, cal_neg = [], []

        with torch.no_grad():
            for batch_idx, (recompressed, sample, labels) in enumerate(Prediction_Loader):
                recompressed = recompressed.to(device)
                sample = sample.to(device)
                labels = labels.long().to(device)

                _ = model(recompressed, sample)

                # feat_ori = cam_ori.feat  # (B, T, C)
                # feat_cal = cam_cal.feat
                feat_ori = model.OriginalBackbone[-1].pre_norm_feat  # (B, T, C)
                feat_cal = model.CalibrationBackbone[-1].pre_norm_feat

                for i in range(labels.size(0)):
                    if labels[i] == 1 and count_pos < MAX_SAMPLES:  # 正样本
                        ori_pos.append(feat_ori[i].reshape(-1))
                        cal_pos.append(feat_cal[i].reshape(-1))
                        count_pos += 1
                    elif labels[i] == 0 and count_neg < MAX_SAMPLES:  # 负样本
                        ori_neg.append(feat_ori[i].reshape(-1))
                        cal_neg.append(feat_cal[i].reshape(-1))
                        count_neg += 1

                if (batch_idx % 1) == 0:
                    print()
                    print(f'[INFO] 测试到第{batch_idx}/{9600 / args.batch_size}个Batch')
                if batch_idx == 5:
                    break

        ori_pos = torch.cat(ori_pos).cpu().numpy()
        ori_neg = torch.cat(ori_neg).cpu().numpy()
        cal_pos = torch.cat(cal_pos).cpu().numpy()
        cal_neg = torch.cat(cal_neg).cpu().numpy()

        plt.figure(figsize=(8, 8))
        plt.hist(ori_neg, bins=100, alpha=0.6, label="Cover")
        plt.hist(ori_pos, bins=100, alpha=0.6, label="Stego")
        # plt.xlabel("Feature")
        # plt.ylabel("Frequency")
        # plt.xticks([])
        # plt.yticks([])
        plt.xlim(-2, 2)  # 你可以根据实际调
        plt.title("Original Backbone Feature Distribution")
        plt.legend(frameon=False)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.hist(cal_neg, bins=100, alpha=0.6, label="Cover")
        plt.hist(cal_pos, bins=100, alpha=0.6, label="Stego")
        # plt.xlabel("Feature")
        # plt.ylabel("Frequency")
        # plt.xticks([])
        # plt.yticks([])
        plt.xlim(-2, 2)
        plt.title("Calibration Backbone Feature Distribution")
        plt.legend(frameon=False)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.show()
