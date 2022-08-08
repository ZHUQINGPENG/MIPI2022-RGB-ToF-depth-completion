import torch
import torch.nn as nn

__all__ = ['conv_bn_relu', 'InvertedResidual', 
            'buildblock_InvertedResidual', 'conv_bn_relu_up']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_bn_relu(in_ch, out_ch, ksize, stride, bn=True, relu=True):
    block = []
    block.append(nn.Conv2d(in_ch, out_ch, ksize, stride, 1, bias=not bn))
    if bn:
        block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())    
    return nn.Sequential(*block)


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hid_ch = round(in_ch * expand_ratio)
        self.identity = stride == 1 and in_ch == out_ch

        if expand_ratio == 1:
            self.block = nn.Sequential(
                # dw
                nn.Conv2d(hid_ch, hid_ch, 3, stride, 1, groups=hid_ch, bias=False),
                nn.BatchNorm2d(hid_ch),
                nn.ReLU(),
                # pw-linear
                nn.Conv2d(hid_ch, out_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:            
            self.block = nn.Sequential(
                # pw
                nn.Conv2d(in_ch, hid_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hid_ch),
                nn.ReLU(),
                # dw
                nn.Conv2d(hid_ch, hid_ch, 3, stride, 1, groups=hid_ch, bias=False),
                nn.BatchNorm2d(hid_ch),
                nn.ReLU(),
                # pw-linear
                nn.Conv2d(hid_ch, out_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        if in_ch != out_ch and stride == 1:
            self.res = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                    nn.BatchNorm2d(out_ch))
        else:
            self.res = None

    def forward(self, x):
        if self.identity:
            return x + self.block(x)
        else:
            return self.block(x)


def buildblock_InvertedResidual(in_ch, out_ch, n, t=6, s=1):
    out_ch = _make_divisible(out_ch, 8)
    block = []
    for i in range(n):
        block.append(InvertedResidual(in_ch, out_ch, s if i == 0 else 1, t))
        in_ch = out_ch

    return nn.Sequential(*block)


def conv_bn_relu_up(in_ch, out_ch, ksize, stride=1, padding=1, output_padding=0,
                  bn=True, relu=True):
    assert (ksize % 2) == 1, 'only odd kernel is supported but kernel = {}'.format(ksize)

    block = []
    block.append(nn.Conv2d(in_ch, out_ch, ksize, stride, padding, bias=not bn, padding_mode='reflect'))
    if bn:
        block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    block.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
    return nn.Sequential(*block)