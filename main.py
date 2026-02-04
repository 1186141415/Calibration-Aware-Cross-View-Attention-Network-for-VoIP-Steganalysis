import argparse
import ast

from utils.PlotCAM import Activation
from utils.PlotHIST import PlotHIST
from utils.PlotCOSINE import PlotCOSINE
from utils import set_seed, save_model, save_checkpoint
from data.data_loaders import Traindataloaders, Valdataloaders, Predictionloaders
import torch
import torch.optim as optim

from models.models import BIEN

import torch.nn as nn

import os
from run import train, val, prediction


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:0")

    if args.train:
        train_Loader = Traindataloaders(args)
        val_Loader = Valdataloaders(args)

        # 初始化
        model = BIEN(float(args.length)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 0.000002
        CLoss = nn.CrossEntropyLoss().to(device)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.2,
            patience=5,
            min_lr=8e-7,
            threshold=1e-4,
            verbose=True
        )

        base_epoch = 0
        best_acc = 0.0
        if args.continue_train:
            # 加载数据
            checkpoint = torch.load(os.path.join(args.model_path, args.model_weight))
            print('load check from :', os.path.join(args.model_path, args.model_weight))
            #################################################################
            # state_dict = checkpoint['model_state_dict']
            # if 'position_embedding.pe' in state_dict:
            #    source_param = state_dict['position_embedding.pe']
            #    if source_param.shape[0] == 35:  # 验证源维度
            #        # 只取前30个位置编码
            #        state_dict['position_embedding.pe'] = source_param[:30, :, :]
            #################################################################
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            base_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']

        for epoch in range(args.epoch):
            # 满足结束条件结束
            if (epoch + base_epoch) == args.epoch:
                break

            # 训练模型
            model.train()
            epoch_loss = train(args, model, optimizer, CLoss, train_Loader, device)
            print(f"Epoch {epoch + base_epoch}/{args.epoch}, Loss: {epoch_loss:.4f}")

            # 评估模型
            model.eval()
            accuracy = val(args, model, val_Loader, device)
            scheduler.step(accuracy)
            print(f"Validation Accuracy: {accuracy:.4f}")

            # 如果效果比较好则保存检查点
            is_best = accuracy > best_acc
            best_acc = max(accuracy, best_acc)
            save_checkpoint(epoch + base_epoch, model.state_dict(), optimizer.state_dict(), epoch_loss, best_acc,
                            prefix=args.model_path)
            if is_best:
                print(f"Find best Accuracy: {best_acc:.4f}")
                # 保存检查点文件
                save_model(epoch + base_epoch, model.state_dict(), optimizer.state_dict(), epoch_loss,
                           prefix=args.model_path)
                f = open(os.path.join(args.model_path, "result.txt"), 'a')
                f.write("loaded best_checkpoint (epoch %d, best_acc %.4f)\n" % (epoch + base_epoch, best_acc))
                f.close()

    else:
        Prediction_Loader = Predictionloaders(args)

        # 创建模型
        model = BIEN(float(args.length)).to(device)

        # 加载数据
        best_checkpoint = torch.load(os.path.join(args.model_path, args.model_weight))
        print('load bestcheck from :', os.path.join(args.model_path, args.model_weight))
        model.load_state_dict(best_checkpoint['model_state_dict'])

        if args.Activation:  # 绘制激活图
            Activation(args, Prediction_Loader, model, device)

        if args.Hist:
            PlotHIST(args, Prediction_Loader, model, device)

        if args.Cosine:
            PlotCOSINE(args, Prediction_Loader, model, device)

        else:
            # 测试模型
            model.eval()
            accuracy = prediction(args, model, Prediction_Loader, device)
            print(f"Test Accuracy: {accuracy:.4f}")

            # 保存测试结果
            f = open(os.path.join(args.model_path, "test_result.txt"), 'a')
            f.write("test acc %.4f\n" % (accuracy))
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wnet')  # Ternary Information Extraction Network for VoIP Steganalysis

    parser.add_argument('--method', type=str, default='Miao_enta4',  # Geiser
                        help='Target steganography method, option: Geiser, Miao/enta1, Miao/enta2, Miao/enta4.')
    parser.add_argument('--language', type=str, default='Chinese',  # English Chinese
                        help='Language of data')

    parser.add_argument('--mode', type=str, default='sm_length',
                        help='Comparison of em_rate or sample_length(sm_length)')
    parser.add_argument('--length', type=float, default=0.2,
                        help='Sample length, option: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0.')
    parser.add_argument('--em_rate', type=int, default=100,
                        help='Embedding rate, option: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, RAND.')

    parser.add_argument('--model_path', type=str, default='./modelWeight/Miao_enta4/Length/2/',
                        help='Path of the model weight, you do not have to set it when using our trained weights.')
    parser.add_argument('--spilt', type=int, default=5,
                        help='Spilt of data.')
    parser.add_argument('--epoch', type=int, default=40,
                        help='Epoch of training the model.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size during training and testing.')
    parser.add_argument('--seed', type=int, default=666,
                        help='Value of random seed.')

    parser.add_argument('--train', type=ast.literal_eval, default=True,
                        help='Whether to train the model.')
    parser.add_argument('--continue_train', type=ast.literal_eval, default=False,
                        help='Whether continue to train the model.')
    parser.add_argument('--test', type=ast.literal_eval, default=False,
                        help='Whether to test the model.')
    parser.add_argument('--model_weight', type=str, default='epoch_12_best.pth.tar',
                        help='Model weight used for retrain/test')

    # parser.add_argument('--Plot', type=ast.literal_eval, default=True,
    #                    help='Whether to plot figure.')
    parser.add_argument('--TSNE', type=ast.literal_eval, default=False,
                        help='Whether to plot T-SNE figure.')
    parser.add_argument('--TSNETYPE', type=str, default="2D",
                        help='Plot 2D TSNE figure or 3D TSNE figure.')

    parser.add_argument('--Activation', type=ast.literal_eval, default=False,
                        help='Whether to Activation Map.')
    parser.add_argument('--Hist', type=ast.literal_eval, default=False,
                        help='Whether to plot Feature Hist Map.')
    parser.add_argument('--Cosine', type=ast.literal_eval, default=False,
                        help='Whether to plot Cosine distance Map.')
    parser.add_argument('--Single', type=ast.literal_eval, default=False,
                        help='Whether to plot Single sample.')

    args = parser.parse_args()
    main(args)
