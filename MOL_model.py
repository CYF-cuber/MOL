import torch
import torch.nn as nn
from modules.MER_3dCNN import MER_3dCNN
from modules.FCC import FCC
from modules.CCC import edge_feature,get_graph_feature
from modules.Conv_stack import Conv_stack
from modules.LDM_predictor import LDM_predictor
from modules.FlowNet import FlowNet
import argparse

class MOL(nn.Module):
    def __init__(self, args):
        super(MOL, self).__init__()
        self.cls_num = args.cls
        self.neighbor_num = args.neighbor_num
        # for two frames independently
        self.conv_stack = Conv_stack()
        self.FCC_1 = FCC()
        self.FCC_2 = FCC()
        self.CCC_1 = edge_feature()
        self.CCC_2 = edge_feature()
        self.flownet = FlowNet()
        self.mer_3dcnn = MER_3dCNN(cls_num = self.cls_num )
        self.ldm_predictor = LDM_predictor()

        self.dropout = nn.Dropout()
        self.flatten = nn.Flatten()

    def freeze(self,layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        _,_,_,_, video_length = x.shape # [b, 1, 128, 128, 8]
        pred_ldm_list = []
        pred_flow_list = []
        for index in range(video_length - 1):
            frame_1_origin = x[:, :, :, :, index]
            frame_2_origin = x[:, :, :, :, index + 1]

            frame_1 = self.conv_stack(frame_1_origin)
            frame_2 = self.conv_stack(frame_2_origin)

            b,c,h,w = frame_1.size()

            frame_1_feat = self.FCC_1(frame_1)
            frame_2_feat = self.FCC_2(frame_2)

            frame_1_feat_flatten = frame_1_feat.reshape(b, c, h*w )
            frame_2_feat_flatten = frame_2_feat.reshape(b, c, h*w )

            frame_1_neighbor_feat = get_graph_feature(frame_1_feat_flatten, k= self.neighbor_num)
            frame_2_neighbor_feat = get_graph_feature(frame_2_feat_flatten, k= self.neighbor_num)

            frame_1_edge_feature = self.CCC_1(frame_1_feat_flatten,frame_1_neighbor_feat).reshape(b,c,h,w)
            frame_2_edge_feature = self.CCC_2(frame_2_feat_flatten,frame_2_neighbor_feat).reshape(b,c,h,w)
    
            frame_1_feat = frame_1_feat+frame_1_edge_feature
            frame_2_feat = frame_2_feat+frame_2_edge_feature


            if index == 0:
                cross_feat = torch.cat([frame_1_feat.unsqueeze(-1), frame_2_feat.unsqueeze(-1)], dim=-1)
            else:
                cross_feat = torch.cat([cross_feat, frame_1_feat.unsqueeze(-1)], dim=-1)
                cross_feat = torch.cat([cross_feat, frame_2_feat.unsqueeze(-1)], dim=-1)

            pre_ldm = self.ldm_predictor(frame_2_feat).contiguous()
            pred_ldm_list.append(pre_ldm)
            
            pred_flow = self.flownet([frame_1_origin, frame_2_origin], torch.cat((frame_1_feat.contiguous(), frame_2_feat.contiguous()), dim=1))
            pred_flow_list.append(pred_flow)

        pre_mer = self.mer_3dcnn(cross_feat)
        return pre_mer, pred_flow_list, pred_ldm_list

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--CLASS_NUM', type=int, default=5)
    args = parser.parse_args()

    x = torch.ones(32, 1, 128, 128, 8).cuda()
    model = MOL(args).cuda()
    mer, flow, ldm = model(x)
    print(mer.shape)
