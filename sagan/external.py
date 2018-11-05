
import torch
from torch.optim.optimizer import Optimizer, required

import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# (Modified) From https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()

        self.kernel_size = 1

        self.query_conv = nn.utils.spectral_norm(nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=self.kernel_size))
        self.key_conv = nn.utils.spectral_norm(nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=self.kernel_size))
        self.value_conv = nn.utils.spectral_norm(nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=self.kernel_size))
        self.gamma = nn.Parameter(torch.FloatTensor(1).fill_(0))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        px = x
#        px = nn.functional.pad(px, (1, 1, 1, 1), mode='reflect')

        proj_query = self.query_conv(px).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(px).view(
            m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(px).view(
            m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
