'''
계산시 필요한 것들 모아놓은 파일
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# import GPUtil


class BatchNorm(nn.Module):
    def __init__(self, num_features):
        super(BatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features = num_features,  eps = 1e-6)  # num_channels = channels,
        
    def forward(self, x):
        return self.bn(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_feat, out_feat, normalize = True):
        super(ResidualBlock, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        if normalize:
            self.block = nn.Sequential(
                nn.Linear(in_feat, out_feat),
                BatchNorm(out_feat),
                nn.LeakyReLU(0.2, inplace = True),  # Negative slope of the LeakyReLU, Default: 0.01
            )

        else:
            self.block = nn.Sequential(
                nn.Linear(in_feat, out_feat),
                nn.LeakyReLU(0.2, inplace = True),  # Negative slope of the LeakyReLU, Default: 0.01
            )        

        if in_feat != out_feat:
            self.channel_up = nn.Linear(in_feat, out_feat)
        
    def forward(self, x):  # , normalize = True  Residual Block

        if self.in_feat != self.out_feat:  # 다르면 (Group_Norm과 Swish 안한 output) + (다 처리한 output)
            return self.channel_up(x) + self.block(x)
        else:  # in_feat와 out_feat이 같으면 input + block(input)
            return x + self.block(x)



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.nn.LeakyReLU(x)  # 여기서 out_of_memory



# class UpSampleBlock(nn.Module):  # Single convolution layer 
#     def __init__(self, channels):
#         super(UpSampleBlock, self).__init__()
#         self.conv = nn.Embedding(channels, channels, 3, 1, 1)
        
#     def forward(self, x):
        
#         x = F.interpolate(x, scale_factor = 2.0) 

#         return self.conv(x)

# class DownSampleBlock(nn.Module):  # Reverse of the up sample blocks 
#     def __init__(self, channels):
#         super(DownSampleBlock, self).__init__()
#         self.conv = nn.Conv2d(channels, channels, 3, 2, 0)
        
#     def forward(self, x):
#         pad = (0, 1, 0, 1)
#         x = F.pad(x, pad, mode = "constant", value = 0)
#         return self.conv(x)
    
# # Sort of the Attention Mechanism
# class NonLocalBlock(nn.Module):
#     def __init__(self, d_model):
#         super(NonLocalBlock, self).__init__()
#         self.d_model = d_model
        
#         self.gn = BatchNorm(d_model)
#         self.q = nn.Linear(d_model, d_model)
#         self.k = nn.Linear(d_model, d_model)
#         self.v = nn.Linear(d_model, d_model)
#         self.proj_out = nn.Linear(d_model, d_model)
        
#     def forward(self, x):
#         h_ = self.gn(x)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)
        
#         b, c, h, w = q.shape
        
#         q = q.reshape(b, c, h*w)
#         q = q.permute(0, 2, 1)#.to("cpu")
#         k = k.reshape(b, c, h*w)#.to("cpu")
#         v = v.reshape(b, c, h*w)#.to("cpu")
  
#         attn = torch.bmm(q, k)  # q*k  왜 여기서 차원 계산 오류가 뜨는거야 시부레...

#         attn = attn * (int(c)**(-0.5))  # c is dimension
#         attn = F.softmax(attn, dim = 2)
#         attn = attn.permute(0, 2, 1)
        
#         A = torch.bmm(v, attn)
#         A = A.reshape(b, c, h, w)
#         # A = A .to("cuda")
#         return x + A
    