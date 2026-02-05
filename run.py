import torch
from utils.PlotTSNE import PlotTSNE


def train(args, model, optimizer, CLoss, train_loader, device):
    running_loss = 0.0
    total_num = 0
    for batch_idx, (recompressed, sample, lable) in enumerate(train_loader):
        # 判断是对比不同嵌入率还是时间窗口
        if args.mode == 'em_rate':
            # 准备数据
            recompressed = recompressed.to(device)
            sample = sample.to(device)

            # 准备标签
            lable = torch.eye(2).to(device)[lable.unsqueeze(1).long()].squeeze().to(device)

            # 前向传播
            optimizer.zero_grad()
            result = model(recompressed, sample)

            # 计算损失
            loss = CLoss(result, lable)

            # 反向传播
            loss.backward()

            # 根据梯度更新参数
            optimizer.step()

            # 收集并输出模型统计信息
            running_loss += loss.item() * args.batch_size
            total_num += args.batch_size
            if (batch_idx % 10) == 0:
                print()
                print(
                    f'[INFO] 运行到第{batch_idx}/{60800 / args.batch_size}个Batch。当前损失为:{loss}。')

        # 训练不同时间窗口
        else:
            # 准备数据cover, sample
            recompressed = recompressed.to(device)
            sample = sample.to(device)

            # 准备标签
            lable = torch.eye(2).to(device)[lable.unsqueeze(1).long()].squeeze().to(device)

            # 前向传播
            optimizer.zero_grad()
            result = model(recompressed, sample)

            # 计算损失
            loss = CLoss(result, lable)

            # 反向传播
            loss.backward()

            # 根据梯度更新参数
            optimizer.step()

            # 收集并输出模型统计信息
            running_loss += loss.item() * args.batch_size
            total_num += args.batch_size
            if (batch_idx % 10) == 0:
                print()
                print(f'[INFO] 运行到第{batch_idx}/{60800 / args.batch_size}个Batch。当前监督损失:{loss}。')

    epoch_loss = running_loss / total_num
    return epoch_loss


def val(args, model, val_loader, device):
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for batch_idx, (recompressed, sample, lable) in enumerate(val_loader):
            # 判断是对比不同嵌入率还是时间窗口
            if args.mode == 'em_rate':
                # 准备数据
                recompressed = recompressed.to(device)
                sample = sample.to(device)

                # 准备标签
                lable = lable.long().to(device)

                # 前向传播
                result = model(recompressed, sample)

                # 收集输出模型统计信息
                _, predicted = torch.max(result, 1)
                total_preds += args.batch_size
                correct_preds += (predicted == lable).sum().item()
                if (batch_idx % 10) == 0:
                    print()
                    print(
                        f'[INFO] 评估到第{batch_idx}/{9600 / args.batch_size}个Batch。当前正确预测为：{correct_preds}，总预测为:{total_preds},正确率为：{correct_preds / total_preds}。')

            # 评估不同时间窗口
            else:
                # 准备数据
                recompressed = recompressed.to(device)
                sample = sample.to(device)

                # 准备标签
                lable = lable.long().to(device)

                # 前向传播
                result = model(recompressed, sample)

                # 收集输出模型统计信息
                _, predicted = torch.max(result, 1)
                total_preds += args.batch_size
                correct_preds += (predicted == lable).sum().item()
                if (batch_idx % 10) == 0:
                    print()
                    print(
                        f'[INFO] 评估到第{batch_idx}/{9600 / args.batch_size}个Batch。当前正确预测为：{correct_preds}，总预测为:{total_preds},正确率为：{correct_preds / total_preds}。')

    accuracy = correct_preds / total_preds
    return accuracy


def prediction(args, model, Prediction_Loader, device):  # test
    correct_preds = 0
    total_preds = 0

    features = []  # TSNE变量
    labels = []  # TSNE变量

    with torch.no_grad():
        for batch_idx, (recompressed, sample, lable) in enumerate(Prediction_Loader):
            # 准备数据
            recompressed = recompressed.to(device)
            sample = sample.to(device)

            # 准备标签
            lable = lable.long().to(device)

            # 前向传播
            if args.TSNE:
                result, feat = model(recompressed, sample, return_tsne_feat=args.TSNE)
                features.append(feat.cpu())
                labels.append(lable.cpu())
            else:
                result = model(recompressed, sample)

            # 收集输出模型统计信息
            _, predicted = torch.max(result, 1)
            total_preds += args.batch_size
            correct_preds += (predicted == lable).sum().item()
            if (batch_idx % 10) == 0:
                print()
                print(
                    f'[INFO] 测试到第{batch_idx}/{9600 / args.batch_size}个Batch。当前正确预测为：{correct_preds}，总预测为:{total_preds},正确率为：{correct_preds / total_preds}。')

    if args.TSNE:  #绘制TSNE图
        PlotTSNE(args, features, labels)

    accuracy = correct_preds / total_preds
    return accuracy
