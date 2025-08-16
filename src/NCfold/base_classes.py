import os
import os.path as osp
import shutil
import abc

import torch
from torch import nn




class MlpProjector(nn.Module):
    """MLP projection head.
    """

    def __init__(self, n_in, output_size=128):
        """
        Args:
            n_in:
            output_size:
        """
        super(MlpProjector, self).__init__()
        self.dense = nn.Linear(n_in, output_size)
        self.activation = nn.ReLU()
        self.projection = nn.Linear(output_size, output_size)
        self.n_out = output_size

    def get_n_out(self):
        return self.n_out

    def forward(self, embeddings):
        """
        Args:
            embeddings:

        Returns:

        """
        x = self.dense(embeddings)
        x = self.activation(x)
        x = self.projection(x)

        return x
