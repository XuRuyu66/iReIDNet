
#********************  在全局分支中引入CA，在GCP中特征先使用CA，之后求分块与最大池化相加的平均     **********************
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from  SFTCA_Attention import SFTCA
from resnet import resnet50


class RelationModel(nn.Module):
    def __init__(
            self,
            last_conv_stride=1,
            last_conv_dilation=1,
            num_stripes=6,
            local_conv_out_channels=256,
            num_classes=0):
        super(RelationModel, self).__init__()

        self.base = resnet50(
            pretrained=True,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation)
        self.num_stripes = num_stripes
        self.num_classes = num_classes

        self.ca = SFTCA(channel=2048, h=24, w=8)

        self.local_6_conv_list = nn.ModuleList()
        self.local_4_conv_list = nn.ModuleList()
        self.local_2_conv_list = nn.ModuleList()
        self.rest_6_conv_list = nn.ModuleList()
        self.rest_4_conv_list = nn.ModuleList()
        self.rest_2_conv_list = nn.ModuleList()
        self.relation_6_conv_list = nn.ModuleList()
        self.relation_4_conv_list = nn.ModuleList()
        self.relation_2_conv_list = nn.ModuleList()
        self.global_6_max_conv_list = nn.ModuleList()
        self.global_4_max_conv_list = nn.ModuleList()
        self.global_2_max_conv_list = nn.ModuleList()
        self.global_6_rest_conv_list = nn.ModuleList()
        self.global_4_rest_conv_list = nn.ModuleList()
        self.global_2_rest_conv_list = nn.ModuleList()
        self.global_6_pooling_conv_list = nn.ModuleList()
        self.global_4_pooling_conv_list = nn.ModuleList()
        self.global_2_pooling_conv_list = nn.ModuleList()



        for i in range(num_stripes):
            self.local_6_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(4):
            self.local_4_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(2):
            self.local_2_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(num_stripes):
            self.rest_6_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(4):
            self.rest_4_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(2):
            self.rest_2_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        self.global_6_max_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_4_max_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_2_max_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_6_rest_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_4_rest_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_2_rest_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        for i in range(num_stripes):
            self.relation_6_conv_list.append(nn.Sequential(
                nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(4):
            self.relation_4_conv_list.append(nn.Sequential(
                nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(2):
            self.relation_2_conv_list.append(nn.Sequential(
                nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        self.global_6_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_4_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_2_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        if num_classes > 0:
            self.fc_local_6_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_6_list.append(fc)

            self.fc_local_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_4_list.append(fc)

            self.fc_local_2_list = nn.ModuleList()
            for _ in range(2):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_2_list.append(fc)

            self.fc_rest_6_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_rest_6_list.append(fc)

            self.fc_rest_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_rest_4_list.append(fc)

            self.fc_rest_2_list = nn.ModuleList()
            for _ in range(2):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_rest_2_list.append(fc)

            self.fc_local_rest_6_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_6_list.append(fc)

            self.fc_local_rest_4_list = nn.ModuleList()
            for _ in range(4):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_4_list.append(fc)

            self.fc_local_rest_2_list = nn.ModuleList()
            for _ in range(2):
                fc = nn.Linear(local_conv_out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_2_list.append(fc)

            self.fc_global_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_6_list.append(fc)

            self.fc_global_4_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_4_list.append(fc)

            self.fc_global_2_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_2_list.append(fc)

            self.fc_global_max_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_max_6_list.append(fc)

            self.fc_global_max_4_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_max_4_list.append(fc)

            self.fc_global_max_2_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_max_2_list.append(fc)

            self.fc_global_rest_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_rest_6_list.append(fc)

            self.fc_global_rest_4_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_rest_4_list.append(fc)

            self.fc_global_rest_2_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_rest_2_list.append(fc)

    def forward(self, x):

        feat = self.base(x)  # 8  2048 24 8
        ca = self.ca(feat)

        assert (feat.size(2) % self.num_stripes == 0)
        stripe_h_6 = int(feat.size(2) / self.num_stripes)  # 4 分成六个局部特征向量 每个局部特征高度为4 num_stripes=6
        stripe_h_4 = int(feat.size(2) / 4)  # 6   分成四个局部特征向量  每个局部特征高度为6
        stripe_h_2 = int(feat.size(2) / 2)  # 12   分成两个局部特征向量 每个局部特征高度为12
        local_6_feat_list = []
        local_4_feat_list = []
        local_2_feat_list = []  # 计算分成6份、4份、2份时，不同的local特征
        local_6_SFTCA_list = []
        local_4_SFTCA_list = []
        local_2_SFTCA_list = []
        final_feat_list = []
        logits_list = []  # 假设final为256，fc[256,751]==logits表示最终的预测结果
        rest_6_feat_list = []  # one vs rest rest表示除one以外其它五个局部特征整合pooling后的特征
        rest_4_feat_list = []
        rest_2_feat_list = []
        logits_local_rest_list = []
        logits_local_list = []  # 只有p1,p2...p6的
        logits_rest_list = []  # 只有rest的特征
        logits_global_list = []  # local与rest拼接起来的特征
        global_feat_6_list = []
        global_feat_4_list = []
        global_feat_2_list = []




        for i in range(self.num_stripes):  # 得到6块中每一个的特征
            local_6_feat = F.max_pool2d(
                feat[:, :, i * stripe_h_6: (i + 1) * stripe_h_6, :],  # 每一块是4*w
                (stripe_h_6, feat.size(-1)))  # pool成1*1的  maxpooling
            # print(local_6_feat.shape) #8 2048 1 1
            local_6_feat_list.append(local_6_feat)  # 按顺序得到每一块的特征


        for i in range(self.num_stripes):  # 得到6块中每一个的特征
            global_6_feat = (feat[:, :, i * stripe_h_6: (i + 1) * stripe_h_6, :] ) # 每一块是4*w

            # print(local_6_feat.shape) #8 2048 1 1
            global_feat_6_list.append(global_6_feat)  # 按顺序得到每一块的特征

        for i in range(self.num_stripes):  # 得到6块中每一个的特征
            local_SFTCA_6_feat = F.max_pool2d(
                ca[:, :, i * stripe_h_6: (i + 1) * stripe_h_6, :],  # 每一块是4*w
                (stripe_h_6, ca.size(-1)))  # pool成1*1的  maxpooling
            # print(local_6_feat.shape) #8 2048 1 1
            local_6_SFTCA_list.append(local_SFTCA_6_feat)  # 按顺序得到每一块的特征
        global_6_max_feat = ca
        global_6_max_feat = F.max_pool2d(global_6_max_feat, (global_6_max_feat.size(2), global_6_max_feat.size(3)))  # 8 2048 1 1，全局特征
        # print(global_6_max_feat.shape)
        global_6_add_feat = (local_6_SFTCA_list[0] + local_6_SFTCA_list[1] + local_6_SFTCA_list[2]  # 局部与全局的差异
                              + local_6_SFTCA_list[3] + local_6_SFTCA_list[4] + local_6_SFTCA_list[
                                  5] + global_6_max_feat) / 7
        global_6_cut_feat = (local_6_SFTCA_list[0] + local_6_SFTCA_list[1] + local_6_SFTCA_list[2]  # 局部与全局的差异
                              + local_6_SFTCA_list[3] + local_6_SFTCA_list[4] + local_6_SFTCA_list[
                                  5] - global_6_max_feat) / 5
        global_6_max_feat = self.global_6_max_conv_list[0](global_6_max_feat)  # 8 256 1 1
        global_6_rest1_feat = self.global_6_rest_conv_list[0](global_6_add_feat)  # 8 256 1 1
        global_6_rest2_feat = self.global_6_rest_conv_list[0](global_6_cut_feat)  # 8 256 1 1
        global_6_max_rest_feat = self.global_6_pooling_conv_list[0](
            torch.cat((global_6_rest1_feat, global_6_rest2_feat), 1))
        global_6_feat = (global_6_max_feat + global_6_max_rest_feat).squeeze(3).squeeze(2)

        # 开始论文中左图的特征
        for i in range(self.num_stripes):  # 对于每块特征，除去自己之后其他的特征组合在一起 即local rest

            rest_6_feat_list.append((local_6_feat_list[(i + 1) % self.num_stripes]  # 论文公式1处的ri
                                     + local_6_feat_list[(i + 2) % self.num_stripes]
                                     + local_6_feat_list[(i + 3) % self.num_stripes]
                                     + local_6_feat_list[(i + 4) % self.num_stripes]
                                     + local_6_feat_list[(i + 5) % self.num_stripes]) / 5)

        for i in range(4):
            local_4_feat = F.max_pool2d(
                feat[:, :, i * stripe_h_4: (i + 1) * stripe_h_4, :],
                (stripe_h_4, feat.size(-1)))
            # print(local_4_feat.shape)

            local_4_feat_list.append(local_4_feat)
        for i in range(4):  # 得到6块中每一个的特征
            global_4_feat = (feat[:, :, i * stripe_h_4: (i + 1) * stripe_h_4, :] ) # 每一块是4*w

            # print(local_6_feat.shape) #8 2048 1 1
            global_feat_4_list.append(global_4_feat)  # 按顺序得到每一块的特征


        global_4_max_feat = ca
        for i in range(4):  # 得到6块中每一个的特征
            local_SFTCA_4_feat = F.max_pool2d(
                ca[:, :, i * stripe_h_4: (i + 1) * stripe_h_4, :],
                (stripe_h_4, ca.size(-1)))  # pool成1*1的  maxpooling
            local_4_SFTCA_list.append(local_SFTCA_4_feat)  # 按顺序得到每一块的特征
        global_4_max_feat = F.max_pool2d(global_4_max_feat, (global_4_max_feat.size(2), global_4_max_feat.size(3)))
        global_4_add_feat = (local_4_SFTCA_list[0] + local_4_SFTCA_list[1] + local_4_SFTCA_list[2]
                              + local_4_SFTCA_list[3] + global_4_max_feat) / 5

        global_4_cut_feat = (local_4_SFTCA_list[0] + local_4_SFTCA_list[1] + local_4_SFTCA_list[2]  # 局部与全局的差异
                             + local_4_SFTCA_list[3] - global_4_max_feat) / 3

        global_4_max_feat = self.global_4_max_conv_list[0](global_4_max_feat)  # 8 256 1 1
        global_4_rest1_feat = self.global_4_rest_conv_list[0](global_4_add_feat)  # 8 256 1 1
        global_4_rest2_feat = self.global_4_rest_conv_list[0](global_4_cut_feat)  # 8 256 1 1
        global_4_max_rest_feat = self.global_4_pooling_conv_list[0](
            torch.cat((global_4_rest1_feat, global_4_rest2_feat), 1))
        global_4_feat = (global_4_max_feat + global_4_max_rest_feat).squeeze(3).squeeze(2)
        for i in range(4):
            rest_4_feat_list.append((local_4_feat_list[(i + 1) % 4]
                                     + local_4_feat_list[(i + 2) % 4]
                                     + local_4_feat_list[(i + 3) % 4]) / 3)

        for i in range(2):
            local_2_feat = F.max_pool2d(
                feat[:, :, i * stripe_h_2: (i + 1) * stripe_h_2, :],
                (stripe_h_2, feat.size(-1)))
            # print(local_2_feat.shape)
            local_2_feat_list.append(local_2_feat)
        for i in range(2):  # 得到6块中每一个的特征
            global_2_feat = (feat[:, :, i * stripe_h_2: (i + 1) * stripe_h_2, :] ) # 每一块是4*w

            # print(local_6_feat.shape) #8 2048 1 1
            global_feat_2_list.append(global_2_feat)  # 按顺序得到每一块的特征

        global_2_max_feat = ca
        for i in range(2):  # 得到6块中每一个的特征
            local_SFTCA_2_feat = F.max_pool2d(
                ca[:, :, i * stripe_h_2: (i + 1) * stripe_h_2, :],
                (stripe_h_2, ca.size(-1)))
            local_2_SFTCA_list.append(local_SFTCA_2_feat)  # 按顺序得到每一块的特征
        global_2_max_feat = F.max_pool2d(global_2_max_feat, (global_2_max_feat.size(2), global_2_max_feat.size(3)))
        global_2_add_feat = (local_2_feat_list[0] + local_2_feat_list[1] + global_2_max_feat) / 3
        global_2_cut_feat = (local_2_feat_list[0] + local_2_feat_list[1] - global_2_max_feat)

        global_2_max_feat = self.global_2_max_conv_list[0](global_2_max_feat)  # 8 256 1 1
        global_2_rest1_feat = self.global_2_rest_conv_list[0](global_2_add_feat)  # 8 256 1 1
        global_2_rest2_feat = self.global_2_rest_conv_list[0](global_2_cut_feat)  # 8 256 1 1
        global_2_max_rest_feat = self.global_2_pooling_conv_list[0](
            torch.cat((global_2_rest1_feat, global_2_rest2_feat), 1))
        global_2_feat = (global_2_max_feat + global_2_max_rest_feat).squeeze(3).squeeze(2)
        for i in range(2):
            rest_2_feat_list.append((local_2_feat_list[(i + 1) % 2]))

        for i in range(self.num_stripes):

            local_6_feat = self.local_6_conv_list[i](local_6_feat_list[i]).squeeze(3).squeeze(2)  # pi
            input_rest_6_feat = self.rest_6_conv_list[i](rest_6_feat_list[i]).squeeze(3).squeeze(2)  # ri
            input_local_rest_6_feat = torch.cat((local_6_feat, input_rest_6_feat), 1).unsqueeze(2).unsqueeze(3)
            local_rest_6_feat = self.relation_6_conv_list[i](input_local_rest_6_feat)
            local_rest_6_feat = (local_rest_6_feat
                                 + local_6_feat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
            final_feat_list.append(local_rest_6_feat)

            if self.num_classes > 0:
                logits_local_rest_list.append(self.fc_local_rest_6_list[i](local_rest_6_feat))  # 当前local和rest的分类结果
                logits_local_list.append(self.fc_local_6_list[i](local_6_feat))  # 当前local的分类结果
                logits_rest_list.append(self.fc_rest_6_list[i](input_rest_6_feat))  # 当前rest的分类结果

        for i in range(4):

            local_4_feat = self.local_4_conv_list[i](local_4_feat_list[i]).squeeze(3).squeeze(2)
            # print(local_4_feat.shape)
            input_rest_4_feat = self.rest_4_conv_list[i](rest_4_feat_list[i]).squeeze(3).squeeze(2)
            # print(input_rest_4_feat.shape)
            input_local_rest_4_feat = torch.cat((local_4_feat, input_rest_4_feat), 1).unsqueeze(2).unsqueeze(3)
            # print(input_local_rest_4_feat.shape)
            local_rest_4_feat = self.relation_4_conv_list[i](input_local_rest_4_feat)
            # print(local_rest_4_feat.shape)
            local_rest_4_feat = (local_rest_4_feat
                                 + local_4_feat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
            # print(local_rest_4_feat.shape)
            final_feat_list.append(local_rest_4_feat)

            if self.num_classes > 0:
                logits_local_rest_list.append(self.fc_local_rest_4_list[i](local_rest_4_feat))
                logits_local_list.append(self.fc_local_4_list[i](local_4_feat))
                logits_rest_list.append(self.fc_rest_4_list[i](input_rest_4_feat))
        # print(np.array(logits_local_rest_list).shape)  #list的值是10 之前的六个加现在的四个

        for i in range(2):

            local_2_feat = self.local_2_conv_list[i](local_2_feat_list[i]).squeeze(3).squeeze(2)
            # print(local_2_feat.shape)
            input_rest_2_feat = self.rest_2_conv_list[i](rest_2_feat_list[i]).squeeze(3).squeeze(2)
            # print(input_rest_2_feat.shape)
            input_local_rest_2_feat = torch.cat((local_2_feat, input_rest_2_feat), 1).unsqueeze(2).unsqueeze(3)
            # print(input_local_rest_2_feat.shape)
            local_rest_2_feat = self.relation_2_conv_list[i](input_local_rest_2_feat)
            # print(local_rest_2_feat.shape)
            local_rest_2_feat = (local_rest_2_feat
                                 + local_2_feat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)
            # print(local_rest_2_feat.shape)
            final_feat_list.append(local_rest_2_feat)

            if self.num_classes > 0:
                logits_local_rest_list.append(self.fc_local_rest_2_list[i](local_rest_2_feat))
                logits_local_list.append(self.fc_local_2_list[i](local_2_feat))
                logits_rest_list.append(self.fc_rest_2_list[i](input_rest_2_feat))
                # print(np.array(logits_local_rest_list).shape)  #list的值是12 之前的10个加现在的两个

        # 右边图（全局）的所有值6/4/2都要存到list当中
        final_feat_list.append(global_6_feat)
        final_feat_list.append(global_4_feat)
        final_feat_list.append(global_2_feat)


        if self.num_classes > 0:
            logits_global_list.append(self.fc_global_6_list[0](global_6_feat))
            logits_global_list.append(self.fc_global_max_6_list[0](global_6_max_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_rest_6_list[0](global_6_rest1_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_rest_6_list[0](global_6_rest2_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_4_list[0](global_4_feat))
            logits_global_list.append(self.fc_global_max_4_list[0](global_4_max_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_rest_6_list[0](global_4_rest1_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_rest_6_list[0](global_4_rest2_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_2_list[0](global_2_feat))
            logits_global_list.append(self.fc_global_max_2_list[0](global_2_max_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_rest_6_list[0](global_2_rest1_feat.squeeze(3).squeeze(2)))
            logits_global_list.append(self.fc_global_rest_6_list[0](global_2_rest2_feat.squeeze(3).squeeze(2)))
            # print(np.array(logits_global_list).shape)
            return final_feat_list,logits_local_rest_list, logits_local_list, logits_rest_list,logits_global_list
        return final_feat_list  # 分类损失