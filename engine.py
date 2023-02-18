import torch.optim as optim
from model import *
import util
import numpy as np
class trainer():
    def __init__(self, scaler, in_dim, num_nodes, nhid , dropout, lrate, wdecay, device, supports, dgcn_channels, D, out_fea_dim = 1, T = 12, inductive_adp=False):
        self.model = IAGCN(device, num_nodes, dropout, supports=supports, in_dim=in_dim, 
                        residual_channels=nhid, dgcn_channels=dgcn_channels,dilation_channels=nhid, skip_channels=nhid * 8, 
                        end_channels=nhid * 16,out_fea_dim = out_fea_dim, T = T, D = D,inductive_adp=inductive_adp)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, args, input, real, Mf_inputs, supports_batch, Mf_outputs, E, know_mask):
        # input ： BFNT
        # real：   BFNT
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0)) 

        output, interpolation_out = self.model(input, Mf_inputs, supports_batch,train=True,sample_mask=E, know_node = know_mask)
        output, interpolation_out = output.permute(0,2,1,3), interpolation_out.permute(0,2,1,3)
       
        predict = self.scaler.inverse_transform(output)
        imputation = self.scaler.inverse_transform(interpolation_out)
        
        predict_know_set = predict[:,:,list(know_mask),:]
        
        loss = args.pred_loss * (self.loss(predict, real, 0.0) + self.loss(predict_know_set, imputation, 0.0)) + self.loss(imputation, Mf_outputs, 0.0)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(imputation,Mf_outputs,0.0).item()
        rmse = util.masked_rmse(imputation,Mf_outputs,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real, curr_in, supports, y, unknow_set, E, know_node):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output, interpolation_out = self.model(input, curr_in, supports, False,E, know_node)
        output = output.transpose(1,3)
        
        interpolation_out = self.scaler.inverse_transform(interpolation_out)
        interpolation_out = interpolation_out  # BNTV

        o_ = interpolation_out[:,list(unknow_set),]
        y = real.permute(0,2,1,3) # BNFT
        truth_ = y[:,list(unknow_set),]

        o_ = o_.cuda().data.cpu().numpy()
        truth_ = truth_.cuda().data.cpu().numpy()

        RMSE =  util.masked_mse_np(truth_, o_, 0) ** 0.5
        MAE = util.masked_mae_np(truth_, o_, 0)
        MAPE = util.masked_mape_np(truth_,o_, 0)
        
        return RMSE, MAE, MAPE