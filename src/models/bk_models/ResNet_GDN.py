import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3,
                               stride=1,      padding=1, bias=False)

        self.bn1   = nn.BatchNorm2d(num_features=out_ch)
        self.bn2   = nn.BatchNorm2d(num_features=out_ch)

        # projection
        if stride != 1 or in_ch != out_ch:
            self.proj    = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,
                                     stride=stride, bias=False)
            self.bn_proj = nn.BatchNorm2d(num_features=out_ch)
        else:
            self.proj    = None

        self.relu  = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.proj is not None:
            pr = self.proj(x)
            identity = self.bn_proj(pr)

        out = out + identity
        out = self.relu(out)

        return out

class ResNet_GDN(nn.Module):
    def __init__(self, ch_sz, num_classes=2): # 🌟 1. 정답지 개수 기본값을 10에서 2로 변경
        super().__init__()

        self.conv1 = nn.Conv2d(3, ch_sz, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch_sz)

        # Gated Dilated Conv blocks
        self.conv2_f = nn.Conv2d(ch_sz, ch_sz*2, kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv2_g = nn.Conv2d(ch_sz, ch_sz*2, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2     = nn.BatchNorm2d(ch_sz*2)

        self.conv3_f = nn.Conv2d(ch_sz*2, ch_sz*4, kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv3_g = nn.Conv2d(ch_sz*2, ch_sz*4, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn3     = nn.BatchNorm2d(ch_sz*4)

        self.conv4_f = nn.Conv2d(ch_sz*4, ch_sz*8, kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv4_g = nn.Conv2d(ch_sz*4, ch_sz*8, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn4     = nn.BatchNorm2d(ch_sz*8)

        self.resblk1 = ResBlock(ch_sz*2, ch_sz*2)
        self.resblk2 = ResBlock(ch_sz*2, ch_sz*2)
        self.resblk3 = ResBlock(ch_sz*4, ch_sz*4)
        self.resblk4 = ResBlock(ch_sz*4, ch_sz*4)
        self.resblk5 = ResBlock(ch_sz*8, ch_sz*8)
        self.resblk6 = ResBlock(ch_sz*8, ch_sz*8)

        # 🌟 2. 이미지 크기 맞춤용 '깔때기(Adaptive Pooling)' 추가
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        self.fc1  = nn.Linear(ch_sz*32, 512)
        self.fc2  = nn.Linear(512, num_classes) # 🌟 이진 분류(양성/악성)를 위해 최종 출력 수정

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def gated_dilated_conv(self, x, conv_f, conv_g, bn):
        feat = conv_f(x)
        gate = torch.sigmoid(conv_g(x))
        out = feat * gate
        out = bn(out)
        out = self.relu(out)
        return out

    def forward(self, x):               # (b, 3, 32, 32) 혹은 (b, 3, 128, 128) 등

        out = self.conv1(x)             
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)            

        out = self.gated_dilated_conv(
            out, self.conv2_f, self.conv2_g, self.bn2
            )                           


        out = self.resblk1(out)
        out = self.resblk2(out)
        out = self.pool(out)            

        out = self.gated_dilated_conv(
            out, self.conv3_f, self.conv3_g, self.bn3
            )                           

        out = self.resblk3(out)
        out = self.resblk4(out)
        out = self.pool(out)            

        out = self.gated_dilated_conv(
            out, self.conv4_f, self.conv4_g, self.bn4
        )                               
        out = self.resblk5(out)
        out = self.resblk6(out)
        out = self.pool(out)            

        # 🌟 2. 펼치기(view) 직전에 깔때기를 통과시켜 무조건 (2, 2) 크기로 최종 압축합니다.
        out = self.adaptive_pool(out)

        out = out.view(out.shape[0], -1)    # (b, ch_sz*32)로 완벽하게 규격 고정
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out