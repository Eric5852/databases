from lib.pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

LazyBatch=nn.BatchNorm2d
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, Norm_type):
        super(LayerNorm, self).__init__()
        if Norm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
from lib.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
class DConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(DConv2d, self).__init__()

        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn =BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EDEM(nn.Module):
    def __init__(self, in_channels,dilations):
        super(EDEM, self).__init__()
        out_channel = int(in_channels//2)
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
            DConv2d(out_channel, out_channel, 1, padding=0, dilation=dilations[0], BatchNorm=nn.BatchNorm2d),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            LazyBatch(out_channel),
            nn.ReLU(inplace=True))

        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
            DConv2d(out_channel, out_channel, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=nn.BatchNorm2d),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            LazyBatch(out_channel),
            nn.ReLU(inplace=True))

        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
            DConv2d(out_channel, out_channel, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=nn.BatchNorm2d),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            LazyBatch(out_channel),
            nn.ReLU(inplace=True))
        self.p4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1),
            nn.ReLU(),
            DConv2d(out_channel, out_channel, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=nn.BatchNorm2d),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False),
            LazyBatch(out_channel),
            nn.ReLU(inplace=True))

        self.p5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.conv=nn.Conv2d(5*out_channel, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # print(x.shape)
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        p5= self.p5(x)

        return self.conv(torch.cat((p1, p2, p3, p4,p5), dim=1))+x

class MCAM(nn.Module):
    def __init__(self, dim, num_heads, Norm_type ):
        super(MCAM, self).__init__()
        dilations = [1, 6,  12]

        norm_layer = LazyBatch
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm1 = LayerNorm(dim, Norm_type)
        self.out1 =nn.Conv2d(dim, dim, 1, bias=False)
        self.out2 =nn.Conv2d(dim, dim, 1, bias=False)
        self.out3 = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv1 = nn.Conv2d(dim, dim, (1, 5), padding=(0, 2), groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, (5,1), padding=(2,0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, (1, 1), padding=(0, 0), groups=dim)
        self.conv4 = nn.Conv2d(dim, dim, (3, 3), padding=(1, 1), groups=dim)
        self.conv5 = nn.Conv2d(dim, dim, (1,7), padding=(0, 3), groups=dim)
        self.conv6 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.DCon1 = DConv2d(dim, dim, 1, padding=0, dilation=dilations[0], BatchNorm=norm_layer)
        self.DCon2 = DConv2d(dim, dim, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=norm_layer)
        self.DCon3 = DConv2d(dim, dim, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=norm_layer)





    def forward(self, x):
        b, c, h, w = x.shape
        x1 = self.norm1(x)
        x11 = self.DCon1(x)
        x12 = self.DCon2(x)
        x13 = self.DCon3(x)
        out2=x11+x12+x13
        out2 = self.out1(out2)+x

        att1 = self.conv1(x1)
        att2 = self.conv2(x1)
        att3= self.conv3(x1)
        att4 = self.conv4(x1)
        att5 = self.conv5(x1)
        att6 = self.conv6(x1)
        out1 = att1 + att2 + att3+att4+att5+att6
        out1 = self.out2(out1)


        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(out1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        Attention1 = (q1@ k1.transpose(-2, -1))
        Attention1 = Attention1.softmax(dim=-1)
        out3 = (Attention1 @ v1) + q1
        Attention2 = (q2 @ k2.transpose(-2, -1))
        Attention2 = Attention2.softmax(dim=-1)
        out4 = (Attention2 @ v2) + q2
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.out2(out3) + self.out2(out4)+x

        return out




def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.SiLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = LazyBatch(out_channels)
        self.activation =get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)





class CASFM(nn.Module):
    def __init__(self, in_channels, out_channels=None, nb_Conv=2, activation='ReLU'):
        super(CASFM, self).__init__()
        out_channels = out_channels or in_channels // 2
        self.upconv4 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1))

        # Mlp 部分的初始化
        self.mlp_up = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(out_channels, out_channels)
        )

        self.mlp_skip = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(out_channels, out_channels)
        )

        self.nConvs = self._make_nConv(2 * out_channels, out_channels, nb_Conv, activation)

    def forward(self, up, skip_x):
        # 对 up 进行卷积和上采样
        up = self.upconv4(up)
        up = F.interpolate(up, size=skip_x.size()[2:], mode="bilinear", align_corners=True)

        # 使用 MLP 计算 Shared Weight Attention
        Shared_Weight_Attention = self._calculate_shared_weight_attention(up, skip_x)

        # 加权处理 skip 和 up
        att_skip = skip_x * Shared_Weight_Attention
        att_up = up * Shared_Weight_Attention
        x = torch.cat([att_skip, att_up], dim=1)  # 在通道维度上进行拼接
        return self.nConvs(x)

    def _calculate_shared_weight_attention(self, up, skip):
        # 对 up 和 skip 进行平均池化
        up_pool = F.avg_pool2d(up, (up.size(2), up.size(3)), stride=(up.size(2), up.size(3)))
        skip_pool = F.avg_pool2d(skip, (skip.size(2), skip.size(3)), stride=(skip.size(2), skip.size(3)))

        # 使用 MLP 计算加权结果
        up_pool=self._Flatten(up_pool)
        up_mlp = self.mlp_up(up_pool)
        skip_pool= self._Flatten(skip_pool)
        skip_mlp = self.mlp_skip(skip_pool)
        Shared_Weight = (up_mlp + skip_mlp) / 2.0

        # 生成注意力权重并扩展维度以匹配输入
        Shared_Weight_Attention = torch.sigmoid(Shared_Weight).unsqueeze(2).unsqueeze(3).expand_as(skip)
        return Shared_Weight_Attention

    def _Flatten(self, x):
        return x.view(x.size(0), -1)


    def _make_nConv(self, in_channels, out_channels, nb_Conv, activation):
        layers = []
        for _ in range(nb_Conv):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if activation == 'ReLU':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'LeakyReLU':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)


class CCM(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super(CCM, self).__init__()
        self.Channel_Compression_Module= nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Conv2d= nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            LazyBatch(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        self.Conv2d = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            LazyBatch(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )


    def forward(self, x):
        # 循环多次执行 self.outc
        x=self.Channel_Compression_Module(x)

        x =self.Conv2d(x)
        return x
class outclass(nn.Module):
    def __init__(self, in_channels, num_class):
        super(outclass, self).__init__()


        self.Conv2d= nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            LazyBatch(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )
        self.Conv2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=False),
            LazyBatch(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        )
        self.out = nn.Conv2d(in_channels, num_class, kernel_size=1)

    def forward(self, x):
        # 循环多次执行 self.outc


        x =self.Conv2d(x)
        x = self.out(x)
        return x
class MACNet(nn.Module):
    def __init__(self,n_classes=2,in_channels=512):
        super(MACNet,self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = r'D:\MACNet\lib\pvt_v2_b2.pth'

        # self.backbone = pvt_v2_b3()  # [64, 128, 320, 512]
        # path = r'G:\todesk\Polyp-PVT-main\lib\pvt_v2_b3.pth'

        dim9=in_channels//2
        dim8 =int((in_channels/2)/0.8)
        # print(dim8)
        dim7 =in_channels//4
        dim6 =in_channels//8


        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Conv2 = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))


        self.Conv2d1 = nn.Conv2d(dim6+dim8, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.Conv2d2 = nn.Conv2d(dim8, n_classes, kernel_size=(1, 1), stride=(1, 1))
        self.Conv2d3 = nn.Conv2d(dim7, n_classes, kernel_size=(1, 1), stride=(1, 1))







        self.Con9 =CCM(in_channels=512, out_channels=dim9)
        self.Con8 =CCM(in_channels=320, out_channels=dim8)
        self.Con7 = CCM(in_channels=128, out_channels=dim7)
        self.Con6 = CCM(in_channels=64, out_channels=dim6)

        self.Con61 = CCM(in_channels=dim6, out_channels=dim6)
        self.Con91 = CCM(in_channels=dim8, out_channels=dim6)

        self.skip9 = EDEM(dim9 , dilations=[1, 3, 5, 9])
        self.skip8 = EDEM(dim8, dilations=[1, 6, 11, 17])
        self.skip7 = EDEM( dim7 , dilations=[1, 11, 21, 33])
        self.skip6 = EDEM(dim6, dilations=[1,13,27, 53])


        self.MCAM1= MCAM(dim9, 8, Norm_type='BiasFree')
        self.MCAM2 = MCAM(dim7,8, Norm_type='BiasFree')
        self.MCAM3= MCAM(dim6, 8, Norm_type='BiasFree')



        self.CASFM1 = CASFM(dim9,out_channels=dim8, nb_Conv=2) #2
        self.CASFM2 = CASFM(dim8,out_channels=dim7, nb_Conv=3)
        self.CASFM3 = CASFM(in_channels=dim7,out_channels=dim6, nb_Conv=2)

        self.pool= nn.MaxPool2d(2, stride=2)

        self.outc = outclass(in_channels=in_channels // 8, num_class=n_classes)

    def forward(self, input):
        # layer1, layer2=backone(input)

        pvt = self.backbone(input)
        layer6 = pvt[0]
        layer7 = pvt[1]
        layer8 = pvt[2]
        layer9 = pvt[3]

        # print(layer1.shape, layer2.shape, layer3.shape, layer4.shape,layer5.shape)

        layer9 = self.Con9(layer9)
        layer8 = self.Con8(layer8)
        layer7 = self.Con7(layer7)
        layer6 = self.Con6(layer6)

        outlayer6 = F.interpolate(self.Conv2d1(
            torch.cat((F.interpolate(layer8, size=layer6.size()[2:], mode='bilinear', align_corners=True), layer6),
                      dim=1)), size=input.size()[2:], mode='bilinear', align_corners=True)

        layer6 = self.Con61(layer6)


        layer9=self.MCAM1(layer9)


            # layer9 = self.skip9(layer9)
        layer8 = self.skip8(layer8)
        layer7 = self.skip7(layer7)
        layer6 = self.skip6(layer6)




        add1 = self.CASFM1(layer9, layer8)






        add2 = self.CASFM2(add1, layer7)
        add2 = self.MCAM2(add2)

        add3 = self.CASFM3(add2, layer6)
        add3 = self.MCAM3(add3 + F.interpolate(self.Con91(add1), size=add3.size()[2:], mode='bilinear', align_corners=True))


        add1 = F.interpolate(self.Conv2d2(add1), size=input.size()[2:], mode='bilinear', align_corners=True)

        add2 = F.interpolate(self.Conv2d3(add2), size=input.size()[2:], mode='bilinear', align_corners=True)

        out = F.interpolate(self.outc(add3), size=input.size()[2:], mode='bilinear', align_corners=True)

        return outlayer6,add1,add2,out

def compute_speed(model, input_size, device, iteration=100):
    torch.cuda.set_device(device)
    import time
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    model.eval()
    model = model.cuda()

    input = torch.randn(*input_size, device=device)

    for _ in range(1):
        model(input)

    print('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('\033[1;34mElapsed Time: [%.2f s / %d iter\033[0m]' % (elapsed_time, iteration))
    print('\033[1;31mSpeed Time: %.2f ms / iter   FPS: %.2f\033[0m' % (speed_time, fps))
    return speed_time, fps

if __name__ == '__main__':
    model =  MACNet(n_classes=1,in_channels=256).cuda()
    from lib.flops_counter import get_model_complexity_info
    flop, param = get_model_complexity_info(model, (3, 352, 352), as_strings=True, print_per_layer_stat=False)
    compute_speed(model, (1, 3, 352, 352), int(0), iteration=1000)
    print("GFLOPs: {}".format(flop))
    print("Params: {}".format(param))
    input_tensor = torch.randn(1, 3, 264, 264).cuda()
    add2,add1,outlayer9,out = model(input_tensor)
    print(out.size(), outlayer9.size())





