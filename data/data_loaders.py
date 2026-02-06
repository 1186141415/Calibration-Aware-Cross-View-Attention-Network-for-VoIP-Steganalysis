import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from utils import CutMix_Matrix,CutMix
import torch
import random


def Traindataloaders(args):
    """Get dataloader objects

    Args:
        arg: Argument Namespace

    Returns:
        train_loader: train loader

    """
    trainData = np.load('./dataset/data_{}_{}s_{}_train.npy'.format(args.method, args.length, args.em_rate), allow_pickle=True)
    x, re, y = np.asarray([item[0] for item in trainData]), np.asarray([item[1] for item in trainData]), np.asarray([item[2] for item in trainData])

    if args.mode == 'em_rate':
        exchangeDate=np.load('./dataset/data_{}_{}s_RAND_train.npy'.format(args.method, args.length), allow_pickle=True)
        ex, er, ey = np.asarray([item[0] for item in exchangeDate]), np.asarray([item[1] for item in exchangeDate]), np.asarray([item[2] for item in exchangeDate])

        threshold = 0.5  # 50%的概率执行if语句
        for i in range(len(trainData)):
            if random.random() < threshold:
                lam = np.random.beta(0.6, 0.6)
                M = list(CutMix_Matrix(lam))
                x[i] = CutMix(x[i], ex[i], M)
                re[i] = CutMix(re[i], er[i], M)

    re= torch.from_numpy(re).long()
    x= torch.from_numpy(x).long()
    y= torch.from_numpy(y).long()

    dataset = TensorDataset(re, x, y)

    print()
    print('[INFO] read train dataset file')
    print('[INFO] train number: {}.'.format(len(x)))

    train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True, num_workers=4)

    return train_loader

def Valdataloaders(args):
    """Get dataloader objects

    Args:
        arg: Argument Namespace

    Returns:
        train_loader: val loader

    """

    valData = np.load('./dataset/data_{}_{}s_{}_val.npy'.format(args.method, args.length, args.em_rate), allow_pickle=True)
    x, re, y = np.asarray([item[0] for item in valData]), np.asarray([item[1] for item in valData]), np.asarray([item[2] for item in valData])

    if args.mode == 'em_rate':
        exchangeDate=np.load('./dataset/data_{}_{}s_RAND_val.npy'.format(args.method, args.length), allow_pickle=True)
        ex, er, ey = np.asarray([item[0] for item in exchangeDate]), np.asarray([item[1] for item in exchangeDate]), np.asarray([item[2] for item in exchangeDate])

        threshold = 0.5  # 50%的概率执行if语句
        for i in range(len(valData)):
            if random.random() < threshold:
                lam = np.random.beta(0.6, 0.6)
                M = list(CutMix_Matrix(lam))
                x[i] = CutMix(x[i], ex[i], M)
                re[i] = CutMix(re[i], er[i], M)

    re = torch.from_numpy(re).long()
    x = torch.from_numpy(x).long()
    y = torch.from_numpy(y).long()

    dataset = TensorDataset(re, x, y)

    print()
    print('[INFO] read val dataset file')
    print('[INFO] val number: {}.'.format(len(x)))

    val_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True, num_workers=4)

    return val_loader

def Predictionloaders(args):
    """Get dataloader objects

    Args:
        arg: Argument Namespace

    Returns:
        train_loader: train loader
        val_loader: Prediction loader

    """
    PredictionData = np.load('./dataset/data_{}_{}s_{}_test.npy'.format(args.method, args.length, args.em_rate),allow_pickle=True)
    x, re, y = np.asarray([item[0] for item in PredictionData]), np.asarray([item[1] for item in PredictionData]), np.asarray([item[2] for item in PredictionData])

    re = torch.from_numpy(re).long()
    x = torch.from_numpy(x).long()
    y = torch.from_numpy(y).long()

    dataset = TensorDataset(re, x, y)

    print()
    print('[INFO] read prediction dataset file')
    print('[INFO] prediction number: {}.'.format(len(x)))

    Prediction_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    return Prediction_loader








