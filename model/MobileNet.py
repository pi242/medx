import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MobileNet(nn.Module):
    def __init__(self, kernel_size, output_size, dropout_prob=0):
        super().__init__()

        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob

        self.conv1 = BasicConv(
            in_channels=1, out_channels=32, kernel_size=kernel_size, stride=2
        )

        self.conv2 = BasicDepthwiseConv(
            in_channels=32, out_channels=64, kernel_size=kernel_size
        )
        self.conv3 = BasicDepthwiseConv(
            in_channels=64, out_channels=128, kernel_size=kernel_size, stride=2
        )
        self.conv4 = BasicDepthwiseConv(
            in_channels=128, out_channels=128, kernel_size=kernel_size
        )
        self.conv5 = BasicDepthwiseConv(
            in_channels=128, out_channels=256, kernel_size=kernel_size, stride=2
        )
        self.conv6 = BasicDepthwiseConv(
            in_channels=256, out_channels=256, kernel_size=kernel_size
        )
        self.conv7 = BasicDepthwiseConv(
            in_channels=256, out_channels=512, kernel_size=kernel_size, stride=2
        )
        self.conv8 = BasicDepthwiseConv(
            in_channels=512, out_channels=512, kernel_size=kernel_size
        )
        self.conv9 = BasicDepthwiseConv(
            in_channels=512, out_channels=512, kernel_size=kernel_size, stride=2
        )
        self.conv10 = BasicDepthwiseConv(
            in_channels=512, out_channels=512, kernel_size=kernel_size
        )
        self.conv11 = BasicDepthwiseConv(
            in_channels=512, out_channels=512, kernel_size=kernel_size, stride=2
        )
        self.conv12 = BasicDepthwiseConv(
            in_channels=512, out_channels=512, kernel_size=kernel_size
        )
        self.conv13 = BasicDepthwiseConv(
            in_channels=512, out_channels=1024, kernel_size=kernel_size, stride=2
        )
        self.conv14 = BasicDepthwiseConv(
            in_channels=1024, out_channels=1024, kernel_size=kernel_size
        )

        self.denselayer = nn.Linear(1024, output_size)

        ### Weigths initialization
        self.apply(self.init_weights)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        #print("error?: ", int(x.shape[2]))
        x = F.avg_pool1d(x, x.shape[2])

        x = x.view(x.size(0), -1)  # Flatten

        if self.dropout_prob > 0:
            x = F.dropout(x, p=self.dropout_prob, training=self.training)

        x = self.denselayer(x)

        return x

    def init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def print_net_parameters_num(self):

        tot_params = 0
        for name, p in self.named_parameters():
            print(
                "%s \t Num params: %d \t Shape: %s"
                % (name, np.prod(p.data.shape), str(p.data.shape))
            )
            tot_params += np.prod(p.data.shape)
        print("TOTAL NUMBER OF PARAMETERS: ", tot_params)


class BasicDepthwiseConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding="same", stride=1
    ):
        super().__init__()

        self.padding = padding

        ### Padding (same)
        if kernel_size == 1:
            lpad, rpad = (0, 0)
        else:
            lpad = int(kernel_size / 2)
            rpad = int(np.ceil(kernel_size / 2)) - 1

        self.pad = nn.ConstantPad1d((lpad, rpad), 0)

        self.convdw = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            bias=False,
            groups=in_channels,
            stride=stride,
        )
        self.bndw = nn.BatchNorm1d(in_channels)
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1x1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):

        if self.padding == "same":
            x = self.pad(x)
        x = self.convdw(x)
        x = self.bndw(x)
        x = F.relu(x, inplace=True)

        x = self.conv1x1(x)
        x = self.bn1x1(x)
        x = F.relu(x, inplace=True)

        return x


class BasicConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding="same", **kwargs
    ):
        super().__init__()

        self.padding = padding

        ### Padding (same)
        if kernel_size == 1:
            lpad, rpad = (0, 0)
        else:
            lpad = int(kernel_size / 2)
            rpad = int(np.ceil(kernel_size / 2)) - 1

        self.pad = nn.ConstantPad1d((lpad, rpad), 0)

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, bias=False, **kwargs
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        if self.padding == "same":
            x = self.pad(x)
            
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == "__main__":

    net = MobileNet(16, 4)
    net.print_net_parameters_num()
    1 / 0
    from torch.autograd import Variable

    x = Variable(torch.rand(50, 1, 512))
    out = net(x)

    print(x.shape)
    print(out.shape)
