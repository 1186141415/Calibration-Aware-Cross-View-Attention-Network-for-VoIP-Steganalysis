import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def PlotCOSINE(args, Prediction_Loader, model, device):
    if args.Single:
        Predictioniter = iter(Prediction_Loader)
        recompressed, sample, lable = next(Predictioniter)

        recompressed = recompressed.to(device)
        sample = sample.to(device)
        # 准备标签
        lable = lable.long().to(device)

        _ = model(recompressed, sample)

        R = model.backbone_R[0]
        S = model.backbone_S[0]
        feat = model.backbone_output[0]

        xR = R.transpose(0, 1)  # (C, L)
        xS = S.transpose(0, 1)
        feat = feat.transpose(0, 1)  # (C, L)

        xR = F.normalize(xR, dim=1)
        xS = F.normalize(xS, dim=1)
        feat = F.normalize(feat, dim=1)

        cos_simR = torch.matmul(xR, xR.T)  # (C, C)
        cos_simS = torch.matmul(xS, xS.T)  # (C, C)
        cos_simfeat = torch.matmul(feat, feat.T)  # (C, C)

        cos_distR = 1.0 - cos_simR
        cos_distS = 1.0 - cos_simS
        cos_distfeat = 1.0 - cos_simfeat

        plt.figure(figsize=(10, 8))
        plt.imshow(cos_distR[0:256, 0:256].detach().cpu().numpy(), aspect='auto', cmap='plasma')
        plt.colorbar()
        plt.title("Channel Cosine Distance of Calibration Backbone")
        plt.xlabel("Channel")
        plt.ylabel("Channel")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(cos_distS[0:256, 0:256].detach().cpu().numpy(), aspect='auto', cmap='plasma')
        plt.colorbar()
        plt.title("Channel Cosine Distance of Original Backbone")
        plt.xlabel("Channel")
        plt.ylabel("Channel")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(cos_distfeat.detach().cpu().numpy(), aspect='auto', cmap='plasma')
        plt.colorbar()
        plt.title("Channel Cosine Distance of Backbone")
        plt.xlabel("Channel")
        plt.ylabel("Channel")
        plt.tight_layout()
        plt.show()
    else:
        R = []
        S = []
        feat = []

        MAX_SAMPLES = 3  # 最多五个样本
        count_pos = 0
        count_neg = 0

        for batch_idx, (recompressed, sample, labels) in enumerate(Prediction_Loader):
            recompressed = recompressed.to(device)
            sample = sample.to(device)
            labels = labels.long().to(device)

            _ = model(recompressed, sample)

            for i in range(labels.size(0)):
                if labels[i] == 1 and count_pos < MAX_SAMPLES:  # 正样本
                    R.append(model.backbone_R.cpu())
                    S.append(model.backbone_S.cpu())
                    feat.append(model.backbone_output.cpu())
                    count_pos += 1
                elif labels[i] == 0 and count_neg < MAX_SAMPLES:  # 负样本
                    R.append(model.backbone_R.cpu())
                    S.append(model.backbone_S.cpu())
                    feat.append(model.backbone_output.cpu())
                    count_neg += 1

            if (batch_idx % 1) == 0:
                print()
                print(f'[INFO] 测试到第{batch_idx}/{9600 / args.batch_size}个Batch')
            if batch_idx == 2:
                break

        R = torch.cat(R, dim=0)
        S = torch.cat(S, dim=0)
        feat = torch.cat(feat, dim=0)

        R = R.mean(dim=0)
        S = S.mean(dim=0)
        feat = feat.mean(dim=0)

        xR = R.transpose(0, 1)  # (C, L)
        xS = S.transpose(0, 1)
        feat = feat.transpose(0, 1)  # (C, L)

        xR = F.normalize(xR, dim=1)
        xS = F.normalize(xS, dim=1)
        feat = F.normalize(feat, dim=1)

        cos_simR = torch.matmul(xR, xR.T)  # (C, C)
        cos_simS = torch.matmul(xS, xS.T)  # (C, C)
        cos_simfeat = torch.matmul(feat, feat.T)  # (C, C)

        cos_distR = 1.0 - cos_simR
        cos_distS = 1.0 - cos_simS
        cos_distfeat = 1.0 - cos_simfeat

        plt.figure(figsize=(10, 8))
        plt.imshow(cos_distR[0:256, 0:256].detach().cpu().numpy(), aspect='auto', cmap='plasma')
        plt.colorbar()
        plt.title("Channel Cosine Distance of Calibration Backbone")
        plt.xlabel("Channel")
        plt.ylabel("Channel")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(cos_distS[0:256, 0:256].detach().cpu().numpy(), aspect='auto', cmap='plasma')
        plt.colorbar()
        plt.title("Channel Cosine Distance of Original Backbone")
        plt.xlabel("Channel")
        plt.ylabel("Channel")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.imshow(cos_distfeat.detach().cpu().numpy(), aspect='auto', cmap='plasma')
        plt.colorbar()
        plt.title("Channel Cosine Distance of Backbone")
        plt.xlabel("Channel")
        plt.ylabel("Channel")
        plt.tight_layout()
        plt.show()
