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
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--out_fea_dim',type=int,default=1,help='out_fea_dim')

parser.add_argument('--pred_loss',type=float,default=0.1,help="预测值的loss权重")
parser.add_argument('--near_loss',type=float,default=0.1,help="预测值和插补值接近的loss权重")
parser.add_argument('--inductive_adp',type=bool,default=True,help="是否添加归纳式自适应邻接矩阵")

parser.add_argument('--no_nm',type=int,required=True,help='子图节点数')
parser.add_argument('--n_m',type=int,required=True,help='未知节点数')
parser.add_argument('--seed',type=int,default=0,help='未知节点数')
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

# import hdfs
# def hdfs_download(url, user, hdfs_path, local_path):
#     client = hdfs.client.Client(url, user)
#     return client.download(hdfs_path, local_path)
# os.environ["CUDA_VISIBLE_DEVICES"] ="3"
def main():

    # url = "http://172.31.246.52:50070/"
    # user = 'Weitonglong'
    # hdfs_path='/Weitonglong/data/kriging/' 
    # local_path='./'
    # hdfs_download(url, user,hdfs_path,local_path)
    # print("hdfs_download")
    # dirs = os.listdir("./")
    # print(dirs) 

    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    
    # sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    # --n_o 150 --h 12 --n_m 50 --n_u 50
    
    n_m = args.n_m   # 选择50个节点为不知道的节点
    no_nm = args.no_nm  # metr共207个节点，减去未知节点还剩157个节点，每次采样的时候，从157个节点中采样出150个节点
    A,X,training_set,val_set,test_set,unknow_set,full_set,know_set,A_s = util.load_data(args.dataset, n_m,args.seed)
    
    # 对于训练集，一路要经过上面的预测模块，这里的输入是已知节点的邻接矩阵
    # 对于测试集，全部节点的邻接矩阵都知道了， 因此要算两个
    # A_s: 已知节点组成的邻接矩阵
    # A： 所有节点构成的邻接矩阵
    adj_mx_train = util.load_adj_new(A_s, args.adjtype)  #这里是得到了两个，入矩阵和出矩阵
    adj_mx_val_test = util.load_adj_new(A, args.adjtype)  # 


    # dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    dataloader = util.load_dataset(training_set, val_set, test_set, args.batch_size, args.batch_size, args.batch_size, no_nm, n_m, unknow_set, A_s)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx_train]
    supports_valtest = [torch.tensor(i).to(device) for i in adj_mx_val_test]
    print(args)
    
    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None

    log_file = log_save_path + 'result_%d_%s.log' % (args.expid, time.strftime('%m%d%H%M%S'))
    with open(log_file, 'a') as f:
        f.write("dataset=%s, lr=%f,n_o_n_m=%d, n_m=%d,  batch_size = %d, id=%d\n"
                % (args.dataset, args.learning_rate, no_nm, n_m, args.batch_size, args.expid))




    engine = trainer(scaler, args.in_dim, args.seq_length, A_s.shape[0], args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit, out_fea_dim = args.out_fea_dim, T = args.seq_length, V = A_s.shape[0],
                         inductive_adp = args.inductive_adp)


    print("start training...",flush=True)
    his_MAE =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y, x_i) in enumerate(dataloader['train_loader'].get_iterator()):
            # x: 预测时的输入：BTNF, 32,12,N_sample，F
            # y: 标签值：BTNF, 32,1,N_sample，F
            # x_i: 插补时的输入：BTNF, 32, 13, N_sample, F
            
            # 处于上面一路预测的数据
            trainx = torch.from_numpy(x.astype('float32')).to(device)  # trainx:BFNT, trainy:BFNT
            trainx= trainx.transpose(1, 3)
            trainy = torch.from_numpy(y.astype('float32')).to(device)
            trainy = trainy.transpose(1, 3)

            # 处理下面一路插补的数据

            #第一步，选出已知的节点，从训练集的节点中选择nonm个，构成子图
            # x_i.shape: BTNF
            x_i = x_i.transpose((0,3,2, 1))
            know_mask = set(random.sample(range(0,x_i.shape[2]),no_nm))
            x_input = x_i[:,:,list(know_mask),:] # 选出了已知点
            A_dynamic = A_s[list(know_mask),:][:, list(know_mask)] # 构建每一个batch的子图
            adj_mx_batch = util.load_adj_new(A_dynamic, args.adjtype)
            supports_batch = [torch.from_numpy(i.astype('float32')).to(device) for i in adj_mx_batch]

            # 第二步，在每一个子图中，选择部分节点，视为缺失的节点，将对应的行置0
            # 处理一下本身缺失的数据
            inputs_omask = np.ones(np.shape(x_input))
            inputs_omask[x_input == 0] = 0

            missing_index = np.ones((x_input.shape))
            
            missing_mask = random.sample(range(0,no_nm),n_m) #Masked locations
            missing_index[:, :, missing_mask] = 0

            Mf_inputs = x_input * inputs_omask * missing_index
            Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32')).to(device)        

            #第三步，处理一下当前的输出
            y = y.transpose((0,3,2, 1))
            outputs = y[:,:,list(know_mask),] #输出也是只有已知点的数据
            outputs = torch.from_numpy(outputs.astype('float32')).to(device)
            missing_index = torch.from_numpy(missing_index.astype("float32")).to(device)        

            E = np.ones((no_nm,32))
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

            E = np.zeros((A.shape[0], 32))
            # E[list(know_set),:] = 0
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
    # exit()
    #testing
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
        # outputs.append(preds.squeeze())
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
    # 存储计算结果，真实值数组和目标值数组
    # print('yhat.cpu().numpy()', yhat.cpu().numpy().shape)
    # print('realy.cpu().numpy()', realy.cpu().numpy().shape)
    np.savez_compressed(result_save_path + 'result_' + str(args.expid), predict=o_,
                        target=truth_)
    # -------------------------------

    print("Training finished")
    # print("The valid loss on best model is", str(round(his_loss[bestid],4)))



    # amae = []
    # amape = []
    # armse = []
    # print(yhat.shape)
    # print(realy.shape)
    # for i in range(12):
    #     pred = scaler.inverse_transform(yhat[:,:,:,i])
    #     real = realy[:,:,:,i]
    #     metrics = util.metric(pred,real)
    #     log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    #     print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
    #     amae.append(metrics[0])
    #     amape.append(metrics[1])
    #     armse.append(metrics[2])

    # log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    # print(log.format(np.mean(amae),np.mean(amape),np.mean(armse)))
    torch.save(engine.model.state_dict(), model_save_path+"_exp"+str(args.expid)+"_best_"+str(round(his_MAE[bestid],2))+".pth")



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))

# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --adjdata /data/ZhuYuting/pyspace/baseline/data/data/chengdu/adj_spatial.npy --num_nodes 144 --data data/chengdu12/ --device 'cuda:1' --in_dim 2 --out_fea_dim 2 --save 'chengdu_model/' --result_dir chengdu_result/ --expid 1
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --adjdata /data/ZhuYuting/pyspace/baseline/data/data/metro10min/adj_spatial.npy --num_nodes 80 --data data/metro10min/ --device 'cuda:1' --in_dim 2 --out_fea_dim 2 --save 'metro10min_model/' --result_dir metro10min_result/ --expid 1
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --adjdata /data/ZhuYuting/pyspace/baseline/data/data/toll420/adj_spatial.npy --num_nodes 420 --data data/toll420/ --device 'cuda:1' --in_dim 2 --out_fea_dim 2 --save 'toll420_model/' --result_dir toll420_result/ --expid 1
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --adjdata /data/ZhuYuting/pyspace/baseline/data/data/PEM/PEMS08/adj_spatial.npy --num_nodes 170 --data data/PEMS08/ --device 'cuda:1' --in_dim 1 --out_fea_dim 1 --save 'PEMS08_model/' --result_dir PEMS08_result/ --expid 1
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --adjdata /data/ZhuYuting/pyspace/baseline/data/data/PEM/PEMS07/adj_spatial.npy --num_nodes 883 --data data/PEMS07/ --device 'cuda:1' --in_dim 1 --out_fea_dim 1 --save 'PEMS07_model/' --result_dir PEMS07_result/ --expid 1
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --adjdata /data/ZhuYuting/pyspace/baseline/data/data/PEM/PEMS04/adj_spatial.npy --num_nodes 307 --data data/PEMS04/ --device 'cuda:1' --in_dim 1 --out_fea_dim 1 --save 'PEMS04_model/' --result_dir PEMS04_result/ --expid 1
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --adjdata /data/ZhuYuting/pyspace/baseline/data/data/PEM/PEMS03/adj_spatial.npy --num_nodes 358 --data data/PEMS03/ --device 'cuda:1' --in_dim 1 --out_fea_dim 1 --save 'PEMS03_model/' --result_dir PEMS03_result/ --expid 1


# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --adjdata /data/ZhuYuting/pyspace/baseline/data/data/PEM/PEMS04/adj_spatial.npy --num_nodes 307 --data data/PEMS04_pos/ --device 'cuda:1' --in_dim 2 --out_fea_dim 1 --save 'PEMS04_pos_model/' --result_dir PEMS04_pos_result/ --expid 1 --epochs 1
# python train.py --gcn_bool --adjtype doubletransition --addaptadj  --randomadj --adjdata /data/ZhuYuting/pyspace/baseline/data/data/PEM/PEMS08/adj_spatial.npy --num_nodes 170 --data data/PEMS08_pos/ --device 'cuda:2' --in_dim 2 --out_fea_dim 1 --save 'PEMS08_pos_model/' --result_dir PEMS08_pos_result/ --expid 1