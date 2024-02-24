import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys
import math

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # print("line 14",x.shape, A.shape)
        if A.shape[0]!=A.shape[1]:
            x = torch.einsum('bfnl,bnd->bfdl',(x,A))
        else:
            x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()

        c_in_1 = (order*2+1)*c_in
        c_in = (order*3+1)*c_in
        

        self.mlp = linear(c_in,c_out)
        self.mlp_1 = linear(c_in_1, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        if len(support) == 3:
            h = self.mlp(h)
        elif len(support) == 2:
            h = self.mlp_1(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        return h

class D_GCN(nn.Module):
    """
    Neural network block that applies a diffusion graph convolution to sampled location
    """       
    def __init__(self, in_channels, out_channels, orders=2, activation = 'relu'): 
        """
        :param in_channels: Number of time step.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param order: The diffusion steps.
        """
        super(D_GCN, self).__init__()
        self.orders = orders
        self.activation = activation
        self.num_matrices = 2 * self.orders + 1
        self.Theta1 = nn.Parameter(torch.FloatTensor(in_channels * self.num_matrices,
                                             out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)
        stdv1 = 1. / math.sqrt(self.bias.shape[0])
        self.bias.data.uniform_(-stdv1, stdv1)
        
    def _concat(self, x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)
        
    def forward(self, X, A):
        """
        X: B,N,F
        A_q, A_h : N,N
        """
        batch_size = X.shape[0] # batch_size
        num_node = X.shape[1]
        input_size = X.size(2)  # time_length
        supports = A
        
        x0 = X # (B,N,F)
        x = [x0]
        for support in supports:
            x1 = torch.einsum('bnf,nw->bwf',(x0,support)) #torch.mm(support, x0)
            x.append(x1)
            for k in range(2, self.orders + 1):
                x2 = 2 * torch.einsum('bnf,nw->bwf',(x1,support)) - x0
                x.append(x2)
                x1, x0 = x2, x1
        
        x = torch.stack(x, dim=0) # order , B,N,F

        x = x.permute(1,2,0,3)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size, num_node, input_size * self.num_matrices])         
        x = torch.matmul(x, self.Theta1)  # (batch_size * self._num_nodes, output_size)     
        x += self.bias
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'selu':
            x = F.selu(x)   
            
        return x



class IAGCN(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2, out_fea_dim = 2, T = 12, V = 144, inductive_adp=False):
        super(IAGCN, self).__init__()
        self.out_fea_dim = out_fea_dim
        self.T = T
        self.V = V
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.gconv_interpolation = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.start_conv_interpolation = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1
        
        self.adp_W = nn.Linear(32*13, 32)
        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 32).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(32, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        self.inductive_adp = inductive_adp
        self.adp_A1_1 = D_GCN(32,100,2)
        self.adp_A1_2 = D_GCN(100,100,2)
        self.adp_A1_3 = D_GCN(100,32,2)
        
        # self.adp_A1_w = nn.Linear(100,32)

        self.adp_A2_1 = D_GCN(32,100,2)
        self.adp_A2_2 = D_GCN(100,100,2)
        self.adp_A2_3 = D_GCN(100,32,2)
        # self.adp_A2_w = nn.Linear(100,32)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))
                self.gconv_interpolation.append(gcn(dilation_channels,residual_channels,dropout,support_len=2))


        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=1,
                                    kernel_size=(1,1),
                                    bias=True)

        self.end_conv_interpolation_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_interpolation_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=1,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.GNN1 = nn.ModuleList([ D_GCN(32,100,2) for i in range(14)] )
        self.GNN2 = nn.ModuleList([ D_GCN(100,100,2) for i in range(14)] )
        self.GNN3 = nn.ModuleList([ D_GCN(100,32,2) for i in range(14)] )
        # self.GNN_linear = nn.Linear(100,32)

    def forward(self, input, Mf_inputs, supports_batch, train, sample_mask, know_node):
        # print(input.shape,'...',self.receptive_field) # BFN,T+1
        
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
            # x_interpolation = nn.functional.pad(Mf_inputs,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x_interpolation = Mf_inputs
        
        # 输入层
        x = self.start_conv(x)  # B，F，N，T+1，对特征为做了一个卷积
        x_interpolation = self.start_conv(x_interpolation)

        # 下面两个skip用来进行跳连接
        skip = 0
        skip_interpolation = 0

        # 对于插补的值，先通过图卷积把每一时间片的数据给补出来
        x_tmp = []
        for t in range(x_interpolation.shape[-1]):
            x_t_in = x_interpolation[:,:,:,t].permute(0,2,1)  # BNF
            X_s1 = self.GNN1[t](x_t_in, supports_batch)
            X_s2 = self.GNN2[t](X_s1, supports_batch) + X_s1 #num_nodes, rank
            X_s3 = self.GNN3[t](X_s2, supports_batch)  # B,N,64
            x_tmp.append(X_s3)  # TBNF
        x_interpolation_tmp = torch.stack(x_tmp, dim=0).permute(1,3,2,0) # bfnt
        x_interpolation = x_interpolation_tmp

        # 计算插补的自适应邻接矩阵
        if train:
            # 先选出来当前子图中所有节点对应的E
            E1 = self.nodevec1[list(know_node),:]  # N,F
            E2 = self.nodevec2[:, list(know_node)] # F,N

            E1 = E1 * sample_mask  # N,F
            E2 = E2 * sample_mask.permute(1,0) # F,N
        else:
            
            E1 = sample_mask
            E2 = sample_mask.permute(1,0)
            E1[list(know_node), :] = self.nodevec1
            E2[:, list(know_node)] = self.nodevec2

        E1 = E1.unsqueeze(-1)  # N,F,1
        E2 = E2.unsqueeze(-1) # F,N,1

        E1 = E1.permute(2,0,1) # bnf
        E2 = E2.permute(2,1,0) # bnf 

        interpolation_1_1 = self.adp_A1_1(E1,supports_batch)
        interpolation_1_2 = self.adp_A1_2(interpolation_1_1,supports_batch) + interpolation_1_1
        interpolation_1 = self.adp_A1_3(interpolation_1_2,supports_batch)
        # interpolation_1 = self.adp_A1_w(interpolation_1.permute(0,2,3,1)) #BNTF
        # interpolation_1 = interpolation_1.permute(0,3,1,2)
        
        
        interpolation_2_1 = self.adp_A2_1(E2,supports_batch)
        interpolation_2_2 = self.adp_A2_2(interpolation_2_1, supports_batch) + interpolation_2_1
        interpolation_2 = self.adp_A2_3(interpolation_2_2,supports_batch)
        # interpolation_2 = self.adp_A2_w(interpolation_2.permute(0,2,3,1)) #BNTF
        # interpolation_2 = interpolation_2.permute(0,3,1,2)


        interpolation_1 = interpolation_1.permute(1,2,0).squeeze(-1)  # N,F
        interpolation_2 = interpolation_2.permute(2,1,0).squeeze(-1)  # F,N

        adp_interpolation_A = F.softmax(F.relu(torch.mm(interpolation_1, interpolation_2)), dim=-1)
        
        if self.inductive_adp == True:
            new_interpolation_support = [adp_interpolation_A] + supports_batch
        else:
            new_interpolation_support = supports_batch

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)

            # 对于预测：x, 插补：x_interpolation分别通过TCN和GCN，其中TCN共享，只是输入不一样
            residual = x
            residual_interpolation = x_interpolation
            # print(residual_interpolation.shape)
            # dilated convolution
            if train:
                filter = self.filter_convs[i](residual)
                filter = torch.tanh(filter)
                gate = self.gate_convs[i](residual)
                gate = torch.sigmoid(gate)
                x = filter * gate  # BFNT

            # 插补时，也过一下TCN
            filter = self.filter_convs[i](residual_interpolation)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual_interpolation)
            gate = torch.sigmoid(gate)
            x_interpolation = filter * gate  # BFNT

            
            # parametrized skip connection
            if train:
                s = x
                s = self.skip_convs[i](s)
                try:
                    skip = skip[:, :, :,  -s.size(3):]
                except:
                    skip = 0
                skip = s + skip

            # 插补的跳连接
            s_interpolation = x_interpolation
            s_interpolation = self.skip_convs[i](s_interpolation)
            try:
                skip_interpolation = skip_interpolation[:, :, :,  -s_interpolation.size(3):]
            except:
                skip_interpolation = 0
            skip_interpolation = s_interpolation + skip_interpolation

            # 下面开始预测和插补的GCN

            if train: # 预测时，需要使用全局的邻接矩阵及自适应邻接矩阵
                if self.gcn_bool and self.supports is not None:
                    if self.addaptadj:
                        x = self.gconv[i](x, new_supports)  # X: BFNT
                    else:
                        x = self.gconv[i](x,self.supports)
                else:
                    x = self.residual_convs[i](x)

                x = x + residual[:, :, :, -x.size(3):]
                
                x = self.bn[i](x)
            #  下面是插补的，只有自适应图卷积,图卷积的参数也是共享的
            x_interpolation = self.gconv[i](x_interpolation, new_interpolation_support)
            x_interpolation = x_interpolation + residual_interpolation[:, :, :, -x_interpolation.size(3):]
            x_interpolation = self.bn[i](x_interpolation)

        # 预测的输出层
        if train:
            x = F.relu(skip)
            x = F.relu(self.end_conv_1(x))
            x = self.end_conv_2(x)

        # 插补的输出层
        x_interpolation = F.relu(skip_interpolation)
        x_interpolation = F.relu(self.end_conv_interpolation_1(x_interpolation))
        x_interpolation = self.end_conv_interpolation_2(x_interpolation)
        
        
        x = x.permute(0,2,1,3)
        x_interpolation = x_interpolation.permute(0,2,1,3)
        # x = x.transpose(1, 2).reshape((-1, self.V, self.T, self.out_fea_dim)).transpose(1, 2)
        # x_interpolation = x_interpolation.transpose(1, 2).reshape((-1, self.V, self.T, self.out_fea_dim)).transpose(1, 2)
        return x, x_interpolation





