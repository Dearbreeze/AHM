import torch
import numpy as np
import cv2
from tqdm import tqdm
import State as State
from pixelwise_a3c import *
# from new_networkv4 import *
from AHMN import *
import matplotlib.pyplot as plt
import torch.optim as optim
from Dataloader_Heart import *
import argparse
import Visualizer
from models_unet_1 import UNet

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(3407)
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = True  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

parser = argparse.ArgumentParser()
# hyper-parameters in DQN
parser.add_argument('--BATCH_SIZE', type = int, default = 12 , help = 'batch size when updating dqn')
parser.add_argument('--LR', type = float, default = 2e-4, help='Adam: learning rate')
parser.add_argument('--DIS_LR', type = int, default = 3e-4, help = '')
parser.add_argument('--visual_iter', type=int, default= 1, help='Visualization every X iterations')
parser.add_argument('--GAMMA', type = float, default = 0.95, help='GAMMA')
parser.add_argument('--MAX_EPISODE', type = int, default = 10000, help = 'interval of assigning the target network parameters to evaluate network')
# parser.add_argument('--NUM_ACTIONS', type = int, default = 3, help = 'number of actions')
parser.add_argument('--Data_root', type = str, default = '', help = 'path of dataset')
parser.add_argument('--MOVE_RANGE', type = int, default = 2, help = 'range of actions')
parser.add_argument('--mask_ratio', type = int, default = 0.75, help = 'mask_ratio')
parser.add_argument('--Division_number', type = int, default = 16, help = 'Divide img')
parser.add_argument('--img_size', type = int, default = 320, help = 'img size')
parser.add_argument('--save_path', type = str, default = './checkpoints/', help = 'path of model save')
parser.add_argument('--rs', type = float, default = None, help = 'random rate')
parser.add_argument('--pre_train', type = str, default = None,)
parser.add_argument('--Unet_weigth', type = str, default = './Unet_weight/checkpoint-600.pth')
parser.add_argument('--name', type = str, default = 'RLautoMask_len1_weight0.5_test01', help = 'name')


def main(opt, vis):
    same_seeds(64151)
    model = AHMN(in_channel=1).to(device)

    if opt.pre_train is not None:
        print('============== Loading Pre-train Model ==================')
        model.load_state_dict(torch.load(opt.pre_train),)
        # print(model)

    Unet_model = UNet(1, 1, init_feature_num=32, bilinear=False).to(device)
    Unet_model.load_state_dict(torch.load(opt.Unet_weigth)['model'],)
    Unet_model.eval()
    optimizer = optim.Adam(model.parameters(), lr=opt.LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)
    path = opt.Data_root


    train_img_list, train_anno_list, test_img_list, test_anno_list, = PathList(path)

    TrainDataset = MakeDataset(path+'/Train', train_img_list, train_anno_list, True, opt.img_size)


    print('The number of training data: ' + str(len(train_img_list)))


    TrainLoader = DataLoader(TrainDataset, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=8)


    mask_size = int(opt.img_size * 1.0 / opt.Division_number + 0.5)
    # print(mask_size)
    current_state = State.State((opt.BATCH_SIZE, 1, opt.img_size, opt.img_size), mask_size)
    agent = PixelWiseA3C_InnerState(opt, vis, model, optimizer, opt.BATCH_SIZE, opt.GAMMA)


    for n_epi in tqdm(range(0, 10000), ncols=70):

        for batch_idx, (state, _) in enumerate(tqdm(TrainLoader)):
            # break
            gt = state.clone().numpy()
            # print(l.shape,type(l)).clone()
            # raw_n = np.random.normal(0, opt.sigma, (opt.BATCH_SIZE, 1, opt.img_size, opt.img_size)).astype(l.dtype) / 255
            mask = np.ones((gt.shape[0], 1, mask_size, mask_size), dtype=gt.dtype)   # (1, 1, 16, 16)
            # mask = mask.repeat(l.shape[2] / opt.Division_number, axis=2).repeat(l.shape[3] / opt.Division_number, axis=3)
            current_state.reset(state.numpy(), mask)
            reward = np.zeros(mask.shape, mask.dtype)
            sum_reward = 0

            if n_epi % opt.visual_iter == 0:
                # image = np.asanyarray(l[0].transpose(1, 2, 0) * 255, dtype=np.uint8)
                # image = np.squeeze(image)
                # cv2.imw.
                # rite("ori_img.png", image)
                # cv2.waitKey(1)
                vis.img('ori_img', (torch.from_numpy(gt).data.cpu()[0]))


            action, action_prob = agent.act_and_train(current_state.image, reward, opt.mask_ratio, opt.rs)

            if n_epi % opt.visual_iter == 0:
                # print(action[0])
                # print(action_prob[0])
                # paint_amap(action[0])
                vis.heatmap(action[0].data.squeeze(0), opts=dict(xmin=0, xmax=1 ,colormap='Jet'), win='action_heat_map')
            current_state.step(action)
            with torch.no_grad():
                # previous_image =torch.as_tensor(previous_image).to(device)
                current_image =torch.as_tensor(current_state.image).to(device)
                # recon_pre = Unet_model(previous_image)
                recon_cur = Unet_model(current_image)
            if n_epi % opt.visual_iter == 0:
                # vis.img('recon_pre', (recon_pre.data.cpu()[0]))
                vis.img('recon_cur', (recon_cur.data.cpu()[0]))
            balance_mask = np.zeros_like(gt)
            reward = Recon_loss(gt, recon_cur.detach().cpu().numpy(), mask_size, balance_mask)

            sum_reward += np.mean(reward) * np.power(opt.GAMMA, 0)

            agent.stop_episode_and_train(current_state.image, reward, True)
            scheduler.step()
            vis.plot('total reward', sum_reward)
            print("train total reward {a}".format(a=sum_reward))
        vis.save([opt.name])
        torch.save(model.state_dict(), os.path.join(opt.save_path, '{}_{}_Agent_MODEL.pth'.format(opt.name, n_epi)))



def Recon_loss(y_true, y_pred, mask_size, balance_mask):

    # loss = np.zeros([1, 1, mask_size, mask_size], np.float32)
    # loss_balance = np.zeros([1, 1, mask_size, mask_size], np.float32)

    loss = np.zeros([y_true.shape[0], y_true.shape[1], mask_size, mask_size], np.float32)
    loss_balance = np.zeros([y_true.shape[0], y_true.shape[1], mask_size, mask_size], np.float32)


    for i in range(mask_size):
        for j in range(mask_size):
            ture_patch = y_true[:, :, i * 20:(i + 1) * 20, j * 20:(j + 1) * 20]
            pred_patch = y_pred[:, :, i * 20:(i + 1) * 20, j * 20:(j + 1) * 20]
            balance_patch = balance_mask[:, :, i * 20:(i + 1) * 20, j * 20:(j + 1) * 20]
            loss[:, :, i, j] = (np.square(ture_patch - pred_patch) * 255).sum() / (mask_size * mask_size)
            loss_balance[:, :, i, j] = (np.square(balance_patch - pred_patch) * 255).sum() / (mask_size * mask_size)
    # print(loss_balance.shape)
    loss = loss + 0.5 * loss_balance# * (1-mask)
    return loss


if __name__ == '__main__':
    opt = parser.parse_args()
    vis = Visualizer.Visualizer(opt.name)
    main(opt,vis)
