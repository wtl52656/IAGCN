import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import os
import random
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--dataset',type=str,default='metr',help='data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.0001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--out_fea_dim',type=int,default=1,help='out_fea_dim')
parser.add_argument('--dgcn_channels',type=int,default=32,help='dgcn_dim')
parser.add_argument('--D',type=int,default=32,help='diffusion step')
parser.add_argument('--pred_loss',type=float,default=0.1,help="lamda")
parser.add_argument('--inductive_adp',type=bool,required=True,help="whether to add the adaptive matrix")
parser.add_argument('--no_nu',type=int,required=True,help='the number of N^s')
parser.add_argument('--n_u',type=int,required=True,help='the number of N_u^s')
parser.add_argument('--seed',type=int,default=0,help='select set of unknow set')

args = parser.parse_args()
model_save_path = "model/" + args.dataset + "/"
result_save_path = "result/" + args.dataset + "/"
log_save_path = "logs/" + args.dataset +"/"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path)
if not os.path.exists(log_save_path):
    os.makedirs(log_save_path)

def main():

    device = torch.device(args.device)
    
    n_u = args.n_u   # the number of unknown node
    no_nu = args.no_nu  # the number of sampled node in th subgraph
    A,X,training_set,val_set,test_set,unknow_set,full_set,know_set,A_s = util.load_data(args.dataset, n_u,args.seed)
    
    # A_s: adj for training
    # A： adj for test
    adj_mx_train = util.load_adj_new(A_s, args.adjtype)  
    adj_mx_val_test = util.load_adj_new(A, args.adjtype)   

    dataloader = util.load_dataset(training_set, val_set, test_set, args.batch_size, args.batch_size, args.batch_size, no_nu, n_u, unknow_set, A_s)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx_train]
    supports_valtest = [torch.tensor(i).to(device) for i in adj_mx_val_test]
    print(args)
    

    log_file = log_save_path + 'result_%d_%s.log' % (args.expid, time.strftime('%m%d%H%M%S'))
    with open(log_file, 'a') as f:
        f.write("dataset=%s, lr=%f,n_o_n_m=%d, n_m=%d,  batch_size = %d, id=%d\n"
                % (args.dataset, args.learning_rate, no_nu, n_u, args.batch_size, args.expid))


    engine = trainer(scaler, args.in_dim, A_s.shape[0], args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.dgcn_channels,args.D,
                         out_fea_dim = args.out_fea_dim, T = args.seq_length,
                         inductive_adp = args.inductive_adp)

    print("start training...",flush=True)
    his_MAE =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y, x_i) in enumerate(dataloader['train_loader'].get_iterator()):
            # x: input for the constraing task     BTNF, 32,12,N_sample，F
            # y: label                             BTNF, 32,1,N_sample，F
            # x_i: input for the kriging task      BTNF, 32,13,N_sample, F
            
            # for the constraint task
            trainx = torch.from_numpy(x.astype('float32')).to(device)  # trainx: BFNT, trainy: BFNT
            trainx= trainx.transpose(1, 3)
            trainy = torch.from_numpy(y.astype('float32')).to(device)
            trainy = trainy.transpose(1, 3)

            # for the kriging task

            # generate subgraph
            # x_i.shape: BTNF
            x_i = x_i.transpose((0,3,2, 1))
            know_mask = set(random.sample(range(0,x_i.shape[2]),no_nu))
            x_input = x_i[:,:,list(know_mask),:] 
            A_dynamic = A_s[list(know_mask),:][:, list(know_mask)] 
            adj_mx_batch = util.load_adj_new(A_dynamic, args.adjtype)
            supports_batch = [torch.from_numpy(i.astype('float32')).to(device) for i in adj_mx_batch]

            # set the elements of unknown to 0
            inputs_omask = np.ones(np.shape(x_input))
            inputs_omask[x_input == 0] = 0
            missing_index = np.ones((x_input.shape))
            missing_mask = random.sample(range(0,no_nu),n_u) 
            missing_index[:, :, missing_mask] = 0
            Mf_inputs = x_input * inputs_omask * missing_index
            Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32')).to(device)        
            # label
            y = y.transpose((0,3,2, 1))
            outputs = y[:,:,list(know_mask),] 
            outputs = torch.from_numpy(outputs.astype('float32')).to(device)
            missing_index = torch.from_numpy(missing_index.astype("float32")).to(device)        

            E = np.ones((no_nu,32))
            E[list(missing_mask),:] = 0
            E = torch.from_numpy(E.astype("float32")).to(device)

            metrics = engine.train(args, trainx, trainy[:,:,:,:], Mf_inputs, supports_batch, outputs, E, know_mask)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
                with open(log_file, 'a') as f:
                    f.write('Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}\n'.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]))
        t2 = time.time()
        train_time.append(t2-t1)
        
        #validation
        valid_rmse = []
        valid_mae = []
        valid_mape = []

        s1 = time.time()
        for iter, (x, y, x_i) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.from_numpy(x.astype("float32")).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.from_numpy(y.astype("float32")).to(device)
            testy = testy.transpose(1, 3)
            
            # x_i.shape: BTNF
            x_i = x_i.transpose((0,3,2, 1))
            missing_index = np.ones(np.shape(x_i))
            missing_index[:, :, list(unknow_set),] = 0
            missing_index_s = missing_index

            curr_in = x_i * missing_index_s
            curr_in = torch.from_numpy(curr_in.astype("float32")).to(device)

            E = np.zeros((A.shape[0], args.D))
            E = torch.from_numpy(E.astype("float32")).to(device)
            metrics = engine.eval(testx, testy[:,:,:,:], curr_in, supports_valtest, testy[:,:,:,:], unknow_set, E, know_set)
            valid_rmse.append(metrics[0])
            valid_mae.append(metrics[1])
            valid_mape.append(metrics[2])
            
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        with open(log_file, 'a') as f:
            f.write('Epoch: {:03d}, Inference Time: {:.4f} secs\n'.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_rmse = np.mean(valid_rmse)
        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        his_MAE.append(mvalid_mae)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid RMSE: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_rmse, mvalid_mae, mvalid_mape, (t2 - t1)),flush=True)
        with open(log_file, 'a') as f:
            f.write('Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid RMSE: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Training Time: {:.4f}/epoch\n'.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_rmse, mvalid_mae, mvalid_mape, (t2 - t1)))
        if mvalid_mae == min(his_MAE):
            torch.save(engine.model.state_dict(), model_save_path+"test_best_mae_id_"+ str(args.expid)+".pth")
        
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    with open(log_file, 'a') as f:
        f.write("Average Training Time: {:.4f} secs/epoch\n".format(np.mean(train_time)))
        f.write("Average Inference Time: {:.4f} secs\n".format(np.mean(val_time)))
    
    bestid = np.argmin(his_MAE)
    engine.model.load_state_dict(torch.load(model_save_path+"test_best_mae_id_"+ str(args.expid)+ ".pth"))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy[:,:,:,:].permute(0,2,1,3)
    
    for iter, (x, y, x_i) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.from_numpy(x.astype('float32')).to(device)
        testx = testx.transpose(1,3)
        # x_i.shape: BTNF
        x_i = x_i.transpose((0,3,2, 1))
        missing_index = np.ones(np.shape(x_i))
        missing_index[:, :, list(unknow_set),] = 0
        missing_index_s = missing_index

        curr_in = x_i * missing_index_s
        curr_in = torch.from_numpy(curr_in.astype("float32")).to(device)

        E = np.zeros((A.shape[0], 32))
        E = torch.from_numpy(E.astype('float32')).to(device)

        with torch.no_grad():
            preds, interpolation_out = engine.model(testx, curr_in, supports_valtest, False, E, know_set)
        interpolation_out = scaler.inverse_transform(interpolation_out)
        outputs.append(interpolation_out)

    yhat = torch.cat(outputs, dim=0)
    
    yhat = yhat[:realy.size(0),...]  #T,N,F
    print(yhat.shape)
    print(realy.shape)
    o_ = yhat[:,list(unknow_set),]
    truth_ = realy[:,list(unknow_set),]

    o_ = o_.cuda().data.cpu().numpy()
    truth_ = truth_.cuda().data.cpu().numpy()
    RMSE =  util.masked_mse_np(truth_, o_, 0) ** 0.5
    MAE = util.masked_mae_np(truth_, o_, 0)
    MAPE = util.masked_mape_np(truth_,o_, 0)
    print("##################")
    print("TEST result: ", RMSE, MAE, MAPE)
    print("##################")


    with open(log_file, 'a') as f:
        f.write("TEST result: RMSE = %.4f,  MAE = %.4f,  MAPE = %.4f \n"%(RMSE, MAE, MAPE))
    np.savez_compressed(result_save_path + 'result_' + str(args.expid), predict=o_,
                        target=truth_)
    # -------------------------------

    print("Training finished")

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
