from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch.nn.functional as F
import torch

def PlotTSNE(args,features,labels):
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    if args.TSNETYPE == "2D":
        tsne = TSNE(n_components=2, random_state=42)
        feat_tsne = tsne.fit_transform(features)

        plt.figure(figsize=(7, 7))
        plt.scatter(
            feat_tsne[labels == 0, 0],
            feat_tsne[labels == 0, 1],
            # c='red',
            s=8,
            alpha=0.6,
            label='Negative'
        )

        plt.scatter(
            feat_tsne[labels == 1, 0],
            feat_tsne[labels == 1, 1],
            # c='blue',
            s=8,
            alpha=0.6,
            label='Positive'
        )

        plt.legend(frameon=False)
        # plt.xticks([])
        # plt.xlabel('t-SNE Dim 1')
        plt.title('Embedding Rate={}%'.format(args.em_rate))
        plt.tight_layout()
        plt.show()
        # plt.savefig('tsne_2d.svg')
    else:
        tsne = TSNE(n_components=3, random_state=42)
        feat_tsne = tsne.fit_transform(features)

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            feat_tsne[labels == 0, 0],
            feat_tsne[labels == 0, 1],
            feat_tsne[labels == 0, 2],
            # c='red',
            s=6,
            alpha=0.8,
            edgecolors='black',  # 黑色轮廓
            linewidths=0.1,
            label='Negative'
        )

        ax.scatter(
            feat_tsne[labels == 1, 0],
            feat_tsne[labels == 1, 1],
            feat_tsne[labels == 1, 2],
            # c='blue',
            s=6,
            alpha=0.8,
            edgecolors='black',  # 黑色轮廓
            linewidths=0.1,
            label='Positive'
        )

        ax.legend(frameon=False)
        # ax.set_xticks([])
        # ax.set_xlabel('t-SNE Dim 1')
        plt.title('Embedding Rate={}%'.format(args.em_rate))
        plt.tight_layout()

        ax.view_init(elev=35, azim=50)
        #ax.view_init(elev=35, azim=-130)
        #print(ax.elev, ax.azim)
        plt.show()
        # plt.savefig('tsne_3d.svg')