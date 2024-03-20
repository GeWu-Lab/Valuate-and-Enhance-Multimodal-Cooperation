import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import resnet18

class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, out):
        # output = torch.cat((x, y), dim=1)
        output = self.fc_out(out)
        return output

class AVClassifier(nn.Module):
    def __init__(self, args):
        super(AVClassifier, self).__init__()

        n_classes = args.n_classes

        self.fusion_module = ConcatFusion(output_dim=n_classes)

        self.audio_net = resnet18(modality='audio')
        self.visual_net = resnet18(modality='visual')

    def forward(self, audio, visual, drop = None, drop_arg = None):
        visual = visual.permute(0, 2, 1, 3, 4).contiguous()
        a = self.audio_net(audio)
        v = self.visual_net(visual)

        (_, C, H, W) = v.size()
        B = a.size()[0]
        v = v.view(B, -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)

        a = F.adaptive_avg_pool2d(a, 1)
        v = F.adaptive_avg_pool3d(v, 1)

        a = torch.flatten(a, 1)
        v = torch.flatten(v, 1)

        if drop_arg != None:
            if self.__dict__['training'] and drop_arg.warmup == 0:
                self.p = drop_arg.p
                out, update_flag = self.execute_drop([a, v], self.p)
                self.update = update_flag
                self.update = torch.Tensor(self.update).cuda()
                out = torch.cat((a,v),1)
                out = self.fusion_module(out)
                return a,v,out,self.update

            else:
                out = torch.cat((a,v),1)
                out = self.fusion_module(out)
                # self.update = [1] * B
                return a,v,out
        
        else:
            if drop != None:
                for i in range(len(drop)):
                    if drop[i] == 1:
                        a[i,:] = 0.0
                    elif drop[i] == 2:
                        v[i,:] = 0.0

            out = torch.cat((a,v),1)
            out = self.fusion_module(out)

            return a, v, out
    
    def exec_drop(self, a, v, drop):
        if drop == 'audio':
            ad = torch.zeros_like(a)
            vd = v
        
        else:
            ad = a
            vd = torch.zeros_like(v)
        
        out = torch.cat((ad,vd),1)
        out = self.fusion_module(out)

        return out