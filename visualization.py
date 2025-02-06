import pickle
import glob

from torch.utils.data.dataloader import DataLoader
import torch.distributions.multivariate_normal as torchdist

from utils import *
from metrics import *
from model import TrajectoryModel
import matplotlib.pyplot as plt
import copy
import os
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ["CUDA_VISIBLE_DEVICES"] = '0'



# Coordinate conversion code：real coordinates are converted into image coordinates
def world2image(traj_w, H_inv):
    # Converts points from Euclidean to homogeneous space, by (x, y) → (x, y, 1)
    trajnx = [[0] * traj_w.shape[0] for row in range(traj_w.shape[1])]
    trajny = [[0] * traj_w.shape[0] for row in range(traj_w.shape[1])]

    for j in range(traj_w.shape[1]):
        for i in range(traj_w.shape[0]):
            traj_cs1 = traj_w[i, j]

            traj_homog = np.hstack((traj_cs1, np.ones(1))).T
            traj_cam = np.dot(H_inv, traj_homog)
            traj_uvz = np.transpose(traj_cam / traj_cam[2])

            traj_uvz1 = traj_uvz.astype(int)

            trajny[j][i] = traj_uvz1[0]
            trajnx[j][i] = traj_uvz1[1]

    return trajny, trajnx


# Trajectory drawing code:
# draw historical trajectory (red), future real trajectory (blue) and predicted trajectory (yellow)
# trajnx[numbers, time]         trajny[numbers, time]
def plot_trajectory1(trajnrelx, trajnrely, trajnprex, trajnprey, trajnobsy, trajnobsx, numbs):
    for i in range(numbs):
        plt.plot(trajnobsx[i], trajnobsy[i], "r-", markersize=4, label="Real Trajectory")  # 红色的线为历史的轨迹)
        plt.plot(trajnrelx[i], trajnrely[i], "b-", markersize=4, label="Real Trajectory")  # 蓝色的线为真实的轨迹)
        plt.plot(trajnprex[i], trajnprey[i], "y-", markersize=4, label="Pred Trajectory")  # 黄色的线为预测的轨迹)
        plt.plot(np.append(trajnobsx[i][7], trajnrelx[i][0]), np.append(trajnobsy[i][7], trajnrely[i][0]), "r-", markersize=4,
                 label="Real Trajectory")  #



def test(model, loader_test, KSTEPS=20):

    model.eval()
    raw_data_dict = {}
    ade_bigls = []
    fde_bigls = []

    step =0
    pic_cnt = 0
    for batch in loader_test:
        step+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2])) * torch.eye(
            V_obs.shape[2])
        identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1])) * torch.eye(
            V_obs.shape[1])
        identity_spatial = identity_spatial.cuda()
        identity_temporal = identity_temporal.cuda()
        identity = [identity_spatial, identity_temporal]

        V_pred = model(V_obs, identity, obs_traj)  # A_obs <8, #, #>

        V_pred = V_pred.squeeze()
        V_tr = V_tr.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
        #
        # #For now I have my bi-variate parameters
        # #normx =  V_pred[:,:,0:1]
        # #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        #
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).cuda()
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]

        mvnormal = torchdist.MultivariateNormal(mean,cov)
        #

        # #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}

        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())  # V_x (8, 3, 2)

        V_obss = V_obs[:, :, :, :2]
        V_obss[:, :, :, 0] = V_obs[:,:,:,1]
        V_obss[:, :, :, 1] = V_obs[:,:,:,2]
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obss.data.cpu().numpy().squeeze().copy(), V_x[0,:,:].copy())


        #
        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(), V_x[-1,:,:].copy())


        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []
        #
        #
        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]
        #

        for k in range(KSTEPS):
            # Load scene images of each dataset
            im = np.array(Image.open('ETHUCYkeshihua/tools/zara1.jpg'))
            plt.imshow(im)

            V_pred = mvnormal.sample()

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())

            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                target.append(V_y_rel_to_abs[:,n:n+1,:])
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)
        #
                ade_ls[n].append(ade(pred,target,number_of))
                fde_ls[n].append(fde(pred,target,number_of))


            # Load the H.txt files of each dataset
            H = (np.loadtxt("ETHUCYkeshihua/zara1/H.txt"))
            H_inv = np.linalg.inv(H)  # H_inv (3, 3)


            # # V_x_rel_to_abs(8, 3, 2)   V_y_rel_to_abs(12, 3, 2)    V_pred_rel_to_abs(12, 3, 2)
            numbs = V_x_rel_to_abs.shape[1]


            #### Due to the differences between the ETH and UCY datasets,
            #### the scenarios under the ETH and UCY datasets need to be processed differently.
            # ETH dataset （eth, hotel）
            # trajnrely, trajnrelx = world2image(V_y_rel_to_abs, H_inv)  # TRAJ: Tx2 numpy array
            # trajnprey, trajnprex = world2image(V_pred_rel_to_abs, H_inv)  # TRAJ: Tx2 numpy array
            # trajnobsy, trajnobsx = world2image(V_x_rel_to_abs, H_inv)  # TRAJ: Tx2 numpy array

            # UCY dataset （univ, zara1, zara2）
            trajnrelx, trajnrely = world2image(V_y_rel_to_abs, H_inv)  # TRAJ: Tx2 numpy array
            trajnprex, trajnprey = world2image(V_pred_rel_to_abs, H_inv)  # TRAJ: Tx2 numpy array
            trajnobsx, trajnobsy = world2image(V_x_rel_to_abs, H_inv)  # TRAJ: Tx2 numpy array


            # Draw a trajectory visualization
            # trajnx[numbers, time]         trajny[numbers, time]
            plot_trajectory1(trajnrelx, trajnrely, trajnprex, trajnprey, trajnobsy, trajnobsx, numbs)


            plt.axis("off")
            plt.savefig(
                "./traj_fig/pic_{}.png".format(pic_cnt)
            )
            plt.close()
            pic_cnt += 1
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

    ade_ = sum(ade_bigls)/len(ade_bigls)
    fde_ = sum(fde_bigls)/len(fde_bigls)
    return ade_,fde_,raw_data_dict


def main():

    KSTEPS = 20
    ade_ls = []
    fde_ls = []
    print('Number of samples:', KSTEPS)
    print("*" * 50)
    root_ = './checkpoints/'
    dataset = ['IMGCN-ETHUCY/zara1']

    paths = list(map(lambda x: root_ + x, dataset))

    for feta in range(len(paths)):

        path = paths[feta]
        exps = glob.glob(path)
        print('Model being tested are:', exps)
        for exp_path in exps:
            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            # Data prep
            obs_seq_len = args.obs_len
            pred_seq_len = args.pred_len
            data_set = './dataset/' + args.dataset + '/'

            dset_test = TrajectoryDataset(
                data_set + 'test/',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1)

            loader_test = DataLoader(
                dset_test,
                batch_size=1,  # This is irrelative to the args batch size parameter
                shuffle=False,
                num_workers=1)

            model = TrajectoryModel(embedding_dims=64, number_gcn_layers=1, dropout=0.1,
                                    obs_len=8, pred_len=12, n_tcn=5, out_dims=5).cuda()
            model.load_state_dict(torch.load(model_path))

            os.mkdir("./traj_fig")

            ad_ = 999999
            fd_ = 999999
            print("Testing ....")
            ade_,fde_,raw_data_dict = test(model, loader_test)
            ade_ = min(ade_, ad_)
            fde_ = min(fde_, fd_)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ade:", ade_, " fde:", fde_)

        print("*" * 50)

    print("Avg ADE:", sum(ade_ls) / 5)
    print("Avg FDE:", sum(fde_ls) / 5)


if __name__ == '__main__':
    main()