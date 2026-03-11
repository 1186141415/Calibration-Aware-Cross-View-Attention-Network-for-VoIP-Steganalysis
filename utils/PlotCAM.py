from utils import normalize, plot_cam, GradCAM
import torch


def Activation(args, Prediction_Loader, model, device):
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
        model.train()

        Predictioniter = iter(Prediction_Loader)
        recompressed, sample, lable = next(Predictioniter)

        recompressed = recompressed.to(device)
        sample = sample.to(device)
        # 准备标签
        lable = lable.long().to(device)

        result = model(recompressed, sample)

        target_class = 0
        score = result[0, target_class]

        # ================================
        # backward
        # ================================
        model.zero_grad()
        score.backward()

        # ================================
        # Original Backbone Grad-CAM
        # ================================
        A_ori = cam_ori.feat[0]  # (L, C)
        G_ori = cam_ori.grad[0]  # (L, C)

        alpha_ori = G_ori.mean(dim=0)  # (C,)
        cam_ori_map = torch.relu(A_ori * alpha_ori)
        cam_ori_map = normalize(cam_ori_map)

        # ================================
        # Calibration Backbone Grad-CAM
        # ================================
        A_cal = cam_cal.feat[0]
        G_cal = cam_cal.grad[0]

        alpha_cal = G_cal.mean(dim=0)
        cam_cal_map = torch.relu(A_cal * alpha_cal)
        cam_cal_map = normalize(cam_cal_map)

        plot_cam(cam_ori_map[0:414, 0:256], f"Average Grad-CAM (Original Backbone, GT={target_class})")
        plot_cam(cam_cal_map[0:414, 0:256], f"Average Grad-CAM (Calibration Backbone, GT={target_class})")

    else:
        # model.eval()
        cam_ori = GradCAM(
            model=model,
            target_layer=model.OriginalBackbone[-1]
        )

        cam_cal = GradCAM(
            model=model,
            target_layer=model.CalibrationBackbone[-1]
        )
        sum_cam_ori = None
        sum_cam_cal = None
        count = 0
        target_class = 0  # 你现在设的是 0，就保持一致
        for batch_idx, (recompressed, sample, label) in enumerate(Prediction_Loader):
            recompressed = recompressed.to(device)
            sample = sample.to(device)
            label = label.to(device)

            # 只统计真实隐写体
            if label.item() != target_class:
                continue

            result = model(recompressed, sample)

            score = result[0, target_class]

            # ================================
            # backward
            # ================================
            model.zero_grad()
            score.backward()

            # ================================
            # Original Backbone CAM
            # ================================
            A_ori = cam_ori.feat[0]  # (L, C)
            G_ori = cam_ori.grad[0]

            alpha_ori = G_ori.mean(dim=0)
            cam_ori_map = torch.relu(A_ori * alpha_ori)
            # cam_ori_map = normalize(cam_ori_map)

            # ================================
            # Calibration Backbone CAM
            # ================================
            A_cal = cam_cal.feat[0]
            G_cal = cam_cal.grad[0]

            alpha_cal = G_cal.mean(dim=0)
            cam_cal_map = torch.relu(A_cal * alpha_cal)
            # cam_cal_map = normalize(cam_cal_map)

            # ================================
            # 累加
            # ================================
            if sum_cam_ori is None:
                sum_cam_ori = cam_ori_map.detach()
                sum_cam_cal = cam_cal_map.detach()
            else:
                sum_cam_ori += cam_ori_map.detach()
                sum_cam_cal += cam_cal_map.detach()

            count += 1

            if (batch_idx % 10) == 0:
                print()
                print(f'[INFO] 测试到第{batch_idx}/{9600 / args.batch_size}个Batch')

        avg_cam_ori = sum_cam_ori / count
        avg_cam_cal = sum_cam_cal / count

        # avg_cam_ori = normalize(avg_cam_ori)
        # avg_cam_cal = normalize(avg_cam_cal)

        plot_cam(avg_cam_ori[0:414, 0:256], f"Average Grad-CAM (Original Backbone, GT={target_class})")
        plot_cam(avg_cam_cal[0:414, 0:256], f"Average Grad-CAM (Calibration Backbone, GT={target_class})")
