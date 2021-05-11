# generate body BPS feature
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.cvae import BPS_CVAE

import torch
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BPS_CVAE_Sampler(BPS_CVAE):
    def __init__(self, n_bps=10000, n_bps_feat=1, hsize1=1024,  hsize2=512, eps_d=32):
        super(BPS_CVAE_Sampler, self).__init__( n_bps=n_bps, n_bps_feat=n_bps_feat, hsize1=hsize1,  hsize2=hsize2, eps_d=eps_d)

    def sample_fixed(self, eps, scene_feat):
        x = F.relu(self.bn3(self.fc3(eps)))  # eps_d-->512, [bs, 1, 512]
        x = F.relu(self.bn_resblc3(self.fc_resblc3(torch.cat([x, scene_feat[0]], dim=-1))))  # 1024-->512
        x = self.res_block3(x)  # [bs, 1, 512]
        x = F.relu(self.bn_resblc4(self.fc_resblc4(torch.cat([x, scene_feat[1]], dim=-1))))  # 1024-->512
        x = self.res_block4(x)  # [bs, 1, 512]
        x = F.relu(self.bn4(self.fc4(torch.cat([x, scene_feat[2]], dim=-1))))  # 1024-->1024 [bs, 1, 1024]
        sample = F.relu(self.bn5(self.fc5(x)))  # 1024 --> 10000, [bs, 1, 10000]
        return sample

    def sample(self, batch_size, scene_feat):
        eps = torch.randn([batch_size, self.n_bps_feat, self.eps_d], dtype=torch.float32).to(device)
        x = F.relu(self.bn3(self.fc3(eps)))  # eps_d-->512, [bs, 1, 512]
        x = F.relu(self.bn_resblc3(self.fc_resblc3(torch.cat([x, scene_feat[0]], dim=-1))))  # 1024-->512
        x = self.res_block3(x)  # [bs, 1, 512]
        x = F.relu(self.bn_resblc4(self.fc_resblc4(torch.cat([x, scene_feat[1]], dim=-1))))  # 1024-->512
        x = self.res_block4(x)  # [bs, 1, 512]
        x = F.relu(self.bn4(self.fc4(torch.cat([x, scene_feat[2]], dim=-1))))  # 1024-->1024 [bs, 1, 1024]
        sample = F.relu(self.bn5(self.fc5(x)))  # 1024 --> 10000, [bs, 1, 10000]
        return sample, eps


    def interpolate(self, scene_feat, interpolate_len=5):
        eps_start = torch.randn([1, self.n_bps_feat, self.eps_d], dtype=torch.float32).to(device)
        eps_end = torch.randn([1, self.n_bps_feat, self.eps_d], dtype=torch.float32).to(device)
        eps_list = [eps_start]

        for i in range(interpolate_len):
            cur_eps = eps_start + (i+1) * (eps_end - eps_start) / (interpolate_len+1)
            eps_list.append(cur_eps)
        eps_list.append(eps_end)

        gen_list = []
        for eps in eps_list:
            x = F.relu(self.bn3(self.fc3(eps)))  # eps_d-->512, [bs, 1, 512]
            x = F.relu(self.bn_resblc3(self.fc_resblc3(torch.cat([x, scene_feat[0]], dim=-1))))  # 1024-->512
            x = self.res_block3(x)  # [bs, 1, 512]
            x = F.relu(self.bn_resblc4(self.fc_resblc4(torch.cat([x, scene_feat[1]], dim=-1))))  # 1024-->512
            x = self.res_block4(x)  # [bs, 1, 512]
            x = F.relu(self.bn4(self.fc4(torch.cat([x, scene_feat[2]], dim=-1))))  # 1024-->1024 [bs, 1, 1024]
            sample = F.relu(self.bn5(self.fc5(x)))  # 1024 --> 10000, [bs, 1, 10000]
            gen_list.append(sample)
        return gen_list, eps_list