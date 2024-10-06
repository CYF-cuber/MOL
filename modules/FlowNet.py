import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
#from spatial_correlation_sampler import spatial_correlation_sample as spatial_correlation_sample

try:
    from spatial_correlation_sampler import spatial_correlation_sample as spatial_correlation_sample
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load custom correlation module"
                      "which is needed for FlowNetC", ImportWarning)

__all__ = [
     'flownetc', 'flownetc_bn'
]

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )
def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        print("else")
        print("input size()[2:]={}".format(input.size()[2:]))
        print("target size()[2:]={}".format(target.size()[2:]))
        return input[:, :, :target.size(2), :target.size(3)]

class FlowNet(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True):
        super(FlowNet,self).__init__()

        self.batchNorm = batchNorm
        self.conv1      = conv(self.batchNorm,   1,   64, kernel_size=6, stride=2)
        self.conv2      = conv(self.batchNorm,  64,  128, kernel_size=4, stride=2)
        self.conv3      = conv(self.batchNorm, 128,  256, kernel_size=2, stride=2)
        self.conv_redir = conv(self.batchNorm, 256,   32, kernel_size=1, stride=1)

        self.conv3_1 = conv(self.batchNorm, 473 + 128 * 2,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,128)
        self.deconv1 = deconv(258, 64)
        self.deconv0 = deconv(130, 32)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(258)
        self.predict_flow1 = predict_flow(130)
        self.predict_flow0 = predict_flow(35)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, frame, feat):
        x1 = frame[0]
        x2 = frame[1]

        # frame [1, 128, 128]
        # feature [128, 16, 16]
        out_conv1a = self.conv1(x1) # [64, 64, 64]
        # self.conv1 = conv(self.batchNorm, 1, 64, kernel_size=6, stride=2)
        # print("out_conv1a:{}".format(out_conv1a.shape))
        out_conv2a = self.conv2(out_conv1a) # [128, 32, 32]
        # self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=4, stride=2)
        # print("out_conv2a:{}".format(out_conv2a.shape))
        out_conv3a = self.conv3(out_conv2a) # [256, 16, 16]
         #self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=2, stride=2)

        # print("out_conv3a:{}".format(out_conv3a.shape))

        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)
        # print(out_conv3b.shape)

        out_conv_redir = self.conv_redir(out_conv3a)
        out_correlation = correlate(out_conv3a, out_conv3b) # [441, 16, 16]
        # print("out_correlation:{}".format(out_correlation.shape))

        in_conv3_1 = torch.cat([out_conv_redir, out_correlation, feat], dim=1)

        # print("in_conv3_1:{}".format(in_conv3_1.shape))

        out_conv3 = self.conv3_1(in_conv3_1)
        # print("out_conv3:{}".format(out_conv3.shape))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        # print("out_conv4:{}".format(out_conv4.shape))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        # print("out_conv5:{}".format(out_conv5.shape))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        # print("flow6 shape:{}".format(flow6.shape))
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        # print("flow6_up shape:{}".format(flow6_up.shape))
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        # print("flow5 shape:{}".format(flow5.shape))
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        # print("flow5_up shape:{}".format(flow5_up.shape))
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)


        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        # print("flow4 shape:{}".format(flow4.shape))
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        # print("flow4_up shape:{}".format(flow4_up.shape))
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        # print("flow3 shape:{}".format(flow3.shape))
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2a)
        # print("flow3_up shape:{}".format(flow3_up.shape))
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2a)

        concat2 = torch.cat((out_conv2a,out_deconv2,flow3_up),1)

        # print("concat2 shape:{}".format(concat2.shape))
        flow2 = self.predict_flow2(concat2)
        flow2_up = crop_like(self.upsampled_flow2_to_1(flow2), out_conv1a)
        out_deconv1 = crop_like(self.deconv1(concat2), out_conv1a)

        concat1 = torch.cat((out_conv1a, out_deconv1, flow2_up), 1)
        flow1 = self.predict_flow1(concat1)
        flow1_up = crop_like(self.upsampled_flow1_to_0(flow1), x1)
        out_deconv0 = crop_like(self.deconv0(concat1), x1)

        concat0 = torch.cat((x1, out_deconv0, flow1_up), 1)
        flow0 = self.predict_flow0(concat0)

        # print("flow0 shape:{}".format(flow0.shape))

        '''
        if self.training:
            return flow2,flow3,flow4,flow5,flow6
        else:
            return flow2
        '''
        return flow0

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownetc(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetC(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def flownetc_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetC(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

if __name__ == '__main__':
    x1, x2 = torch.ones(32,1,128,128),torch.ones(32,1,128,128)
    f1, f2 = torch.ones(32,128,16,16),torch.ones(32,128,16,16)
    model = FlowNet()
    y = model([x1,x2], torch.cat((f1,f2), dim=1))
    print(y.shape)