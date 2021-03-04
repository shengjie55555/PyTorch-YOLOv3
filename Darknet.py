import torch
import torch.nn as nn
import numpy as np
from utils import load_one_layer_weights


class Basic_Block(nn.Module):
    def __init__(self, param):
        super(Basic_Block, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_channels=param[0], out_channels=param[1], kernel_size=param[2],
                                           stride=param[3], padding=(param[2] - 1) // 2, bias=param[4]),
                                 nn.BatchNorm2d(param[1]),
                                 nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.seq(x)


class Residual_Block(nn.Module):
    def __init__(self, param_1, param_2, num):
        super(Residual_Block, self).__init__()
        self.seq = nn.Sequential()
        self.num = num
        for i in range(self.num):
            self.seq.add_module(str(2 * i + 0), Basic_Block(param_1))
            self.seq.add_module(str(2 * i + 1), Basic_Block(param_2))

    def forward(self, x):
        for i in range(self.num):
            ori_x = x
            x = self.seq[0 + 2 * i](x)
            x = self.seq[1 + 2 * i](x)
            x = ori_x + x
        return x


class Header_Block(nn.Module):
    def __init__(self, param_1, param_2):
        super(Header_Block, self).__init__()
        self.seq = nn.Sequential(Basic_Block(param_1),
                                 nn.Sequential(nn.Conv2d(in_channels=param_2[0], out_channels=param_2[1],
                                                         kernel_size=param_2[2], stride=param_2[3],
                                                         padding=(param_2[2] - 1) // 2, bias=param_2[4])))

    def forward(self, x):
        x = self.seq(x)
        return x


class darknet(nn.Module):
    def __init__(self, classes):
        super(darknet, self).__init__()
        self.classes = classes
        self.basic_block_1 = Basic_Block((3, 32, 3, 1, False))
        self.basic_block_2 = Basic_Block((32, 64, 3, 2, False))
        self.residual_block_1 = Residual_Block((64, 32, 1, 1, False), (32, 64, 3, 1, False), 1)
        self.basic_block_3 = Basic_Block((64, 128, 3, 2, False))
        self.residual_block_2 = Residual_Block((128, 64, 1, 1, False), (64, 128, 3, 1, False), 2)
        self.basic_block_4 = Basic_Block((128, 256, 3, 2, False))
        self.residual_block_3 = Residual_Block((256, 128, 1, 1, False), (128, 256, 3, 1, False), 8)
        self.basic_block_5 = Basic_Block((256, 512, 3, 2, False))
        self.residual_block_4 = Residual_Block((512, 256, 1, 1, False), (256, 512, 3, 1, False), 8)
        self.basic_block_6 = Basic_Block((512, 1024, 3, 2, False))
        self.residual_block_5 = Residual_Block((1024, 512, 1, 1, False), (512, 1024, 3, 1, False), 4)
        self.basic_block_7 = Basic_Block((1024, 512, 1, 1, False))
        self.basic_block_8 = Basic_Block((512, 1024, 3, 1, False))
        self.basic_block_9 = Basic_Block((1024, 512, 1, 1, False))
        self.basic_block_10 = Basic_Block((512, 1024, 3, 1, False))
        self.basic_block_11 = Basic_Block((1024, 512, 1, 1, False))
        self.header_block_1 = Header_Block((512, 1024, 3, 1, False), (1024, 3*(len(self.classes) + 5), 1, 1, True))
        self.basic_block_12 = Basic_Block((512, 256, 1, 1, False))
        self.up_sampling_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.basic_block_13 = Basic_Block((768, 256, 1, 1, False))
        self.basic_block_14 = Basic_Block((256, 512, 3, 1, False))
        self.basic_block_15 = Basic_Block((512, 256, 1, 1, False))
        self.basic_block_16 = Basic_Block((256, 512, 3, 1, False))
        self.basic_block_17 = Basic_Block((512, 256, 1, 1, False))
        self.header_block_2 = Header_Block((256, 512, 3, 1, False), (512, 3*(len(self.classes) + 5), 1, 1, True))
        self.basic_block_18 = Basic_Block((256, 128, 1, 1, False))
        self.up_sampling_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.basic_block_19 = Basic_Block((384, 128, 1, 1, False))
        self.basic_block_20 = Basic_Block((128, 256, 3, 1, False))
        self.basic_block_21 = Basic_Block((256, 128, 1, 1, False))
        self.basic_block_22 = Basic_Block((128, 256, 3, 1, False))
        self.basic_block_23 = Basic_Block((256, 128, 1, 1, False))
        self.header_block_3 = Header_Block((128, 256, 3, 1, False), (256, 3*(len(self.classes) + 5), 1, 1, True))

    def forward(self, x):
        x = self.basic_block_1(x)
        x = self.basic_block_2(x)
        x = self.residual_block_1(x)
        x = self.basic_block_3(x)
        x = self.residual_block_2(x)
        x = self.basic_block_4(x)
        x = self.residual_block_3(x)
        route1 = x
        x = self.basic_block_5(x)
        x = self.residual_block_4(x)
        route2 = x
        x = self.basic_block_6(x)
        x = self.residual_block_5(x)
        x = self.basic_block_7(x)
        x = self.basic_block_8(x)
        x = self.basic_block_9(x)
        x = self.basic_block_10(x)
        x = self.basic_block_11(x)
        y1 = self.header_block_1(x)
        x = self.basic_block_12(x)
        x = self.up_sampling_1(x)
        x = torch.cat((x, route2), 1)
        x = self.basic_block_13(x)
        x = self.basic_block_14(x)
        x = self.basic_block_15(x)
        x = self.basic_block_16(x)
        x = self.basic_block_17(x)
        y2 = self.header_block_2(x)
        x = self.basic_block_18(x)
        x = self.up_sampling_2(x)
        x = torch.cat((x, route1), 1)
        x = self.basic_block_19(x)
        x = self.basic_block_20(x)
        x = self.basic_block_21(x)
        x = self.basic_block_22(x)
        x = self.basic_block_23(x)
        y3 = self.header_block_3(x)
        return y1, y2, y3

    def load_weights(self, weights_file):
        fp = open(weights_file, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = load_one_layer_weights(weights, self.basic_block_1.seq, 0)
        ptr = load_one_layer_weights(weights, self.basic_block_2.seq, ptr)

        for i in range(1 * 2):
            ptr = load_one_layer_weights(weights, self.residual_block_1.seq[i].seq, ptr)

        ptr = load_one_layer_weights(weights, self.basic_block_3.seq, ptr)

        for i in range(2 * 2):
            ptr = load_one_layer_weights(weights, self.residual_block_2.seq[i].seq, ptr)

        ptr = load_one_layer_weights(weights, self.basic_block_4.seq, ptr)

        for i in range(2 * 8):
            ptr = load_one_layer_weights(weights, self.residual_block_3.seq[i].seq, ptr)

        ptr = load_one_layer_weights(weights, self.basic_block_5.seq, ptr)

        for i in range(2 * 8):
            ptr = load_one_layer_weights(weights, self.residual_block_4.seq[i].seq, ptr)

        ptr = load_one_layer_weights(weights, self.basic_block_6.seq, ptr)

        for i in range(2 * 4):
            ptr = load_one_layer_weights(weights, self.residual_block_5.seq[i].seq, ptr)

        ptr = load_one_layer_weights(weights, self.basic_block_7.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_8.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_9.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_10.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_11.seq, ptr)
        ptr = load_one_layer_weights(weights, self.header_block_1.seq[0].seq, ptr)
        ptr = load_one_layer_weights(weights, self.header_block_1.seq[1], ptr)

        ptr = load_one_layer_weights(weights, self.basic_block_12.seq, ptr)

        ptr = load_one_layer_weights(weights, self.basic_block_13.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_14.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_15.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_16.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_17.seq, ptr)
        ptr = load_one_layer_weights(weights, self.header_block_2.seq[0].seq, ptr)
        ptr = load_one_layer_weights(weights, self.header_block_2.seq[1], ptr)

        ptr = load_one_layer_weights(weights, self.basic_block_18.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_19.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_20.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_21.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_22.seq, ptr)
        ptr = load_one_layer_weights(weights, self.basic_block_23.seq, ptr)
        ptr = load_one_layer_weights(weights, self.header_block_3.seq[0].seq, ptr)
        ptr = load_one_layer_weights(weights, self.header_block_3.seq[1], ptr)
