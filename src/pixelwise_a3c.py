
import copy
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import autograd
from torch.distributions import Categorical
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

class PixelWiseA3C_InnerState():

    def __init__(self, opt, vis, model, optimizer, batch_size, gamma, beta=1e-2,
                 phi=lambda x: x,
                 pi_loss_coef=1.0, v_loss_coef=0.5,
                 average_reward_tau=1e-2,
                 act_deterministically=False,
                 average_entropy_decay=0.999,
                 average_value_decay=0.999):

        # self.shared_model = model
        self.model = model
        self.opt = opt
        self.vis = vis
        self.optimizer = optimizer
        self.batch_size = batch_size

        # self.t_max = t_max   #
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.average_reward_tau = average_reward_tau
        self.act_deterministically = act_deterministically
        self.average_value_decay = average_value_decay
        self.average_entropy_decay = average_entropy_decay
        # self.batch_states = batch_states


        self.past_action_log_prob = None
        self.past_action_entropy = None
        self.past_states = None
        self.past_rewards = None
        self.past_values = None
        self.average_reward = 0

        self.explorer = None

        # Stats
        self.average_value = 0
        self.average_entropy = 0

    """
    异步更新参数
    """
    def sync_parameters(self):
        for m1, m2 in zip(self.model.modules(), self.shared_model.modules()):
            m1._buffers = m2._buffers.copy()
        for target_param, param in zip(self.model.parameters(), self.shared_model.parameters()):
            target_param.detach().copy_(param.detach())
    """
    异步更新梯度
    """
    def update_grad(self, target, source):
        target_params = dict(target.named_parameters())
        # print(target_params)
        for param_name, param in source.named_parameters():
            if target_params[param_name].grad is None:
                if param.grad is None:
                    pass
                else:
                    target_params[param_name].grad = param.grad
            else:
                if param.grad is None:
                    target_params[param_name].grad = None
                else:
                    target_params[param_name].grad[...] = param.grad

    def update(self, statevar):
        # assert self.t_start < self.t
        if statevar is None:
            # mask_size = int(self.opt.img_size * 1.0 / self.opt.Division_number + 0.5)
            # R = torch.zeros(batch_size, 1, mask_size, mask_size).cuda()
            R = 0.
        else:
            _, vout, _ = self.model(statevar)
            R = vout.detach()
        pi_loss = 0
        v_loss = 0


        R *= self.gamma
        R += self.past_rewards  # paper equation (8)
        v = self.past_values
        # print(R.size())
        # print(R.size(),v.size())
        advantage = R - v.detach() # (32, 1, 63, 63)   # paper equation (10)

        # print(advantage.size())
        log_prob = self.past_action_log_prob
        # entropy = self.past_action_entropy[i]

        pi_loss -= log_prob * advantage.detach()  # paper equation (11,16)
        # pi_loss -= self.beta * entropy
        v_loss += (v - R) ** 2 / 2  #  paper equation (9,14)

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef
        print(pi_loss.mean().item())
        print(v_loss.mean().item())
        print("==========")
        total_loss = (pi_loss + v_loss).mean()

        print("loss:", total_loss.item())
        self.vis.plot('total_loss', float(total_loss.item()))
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        # self.update_grad(self.shared_model, self.model)
        # self.sync_parameters()
        #
        # self.past_action_log_prob = None
        # self.past_action_entropy = None
        # self.past_states = None
        # self.past_rewards = None
        # self.past_values = None
        #
        # self.t_start = self.t

    def act_and_train(self, state, reward, mask_ratio, rs):
        statevar = torch.Tensor(state).cuda()
        # print('statevar', statevar.size())  #statevar torch.Size([32, 65, 63, 63])
        self.past_rewards = torch.Tensor(reward).cuda()


        pout, vout = self.model(statevar)

        mask_ori = pout.detach().clone()
        b, c, h, w = mask_ori.size()
        action = torch.zeros_like(mask_ori)
        for k in range(b):
            mask = mask_ori[k].reshape(c, 1, -1)

            mask_number = int(mask.size(-1) * mask_ratio)

            vals, indices = mask.topk(k=mask_number, dim=-1, largest=True, sorted=True)
            # print(vals)
            for i in range(mask_number):
                mask[ :, :, indices[ :, :, i]] = -32

            mask = mask.reshape(c, h, w)
            torch.set_printoptions(threshold=np.inf)
            # print(mask)
            action[k] = torch.where(mask == -32, 0, 1)
        action = action.to(dtype=pout.dtype)

        self.past_action_log_prob = torch.log(pout + 1e-8).cuda()
        # F.stack([- F.sum(self.all_prob * self.all_log_prob, axis=1)], axis=1)
        # self.past_action_entropy[self.t] = entropy.cuda()
        self.past_values = vout  # V


        return action.detach().cpu(), pout.detach().cpu()

    def act(self, state):
        with torch.no_grad():
            statevar = torch.Tensor(state).cuda()
            pout, _, = self.model(statevar)

            p_trans = pout.permute([0, 2, 3, 1])
            # print('p_trans',p_trans.size())  # p_trans torch.Size([32, 63, 63, 9])
            dist = Categorical(p_trans)
            action = dist.sample().detach()

            return action.squeeze(1).detach().cpu()



    def stop_episode_and_train(self, state, reward, done=False):
        self.past_rewards = torch.Tensor(reward).cuda()
        batch_size = state.shape[0]
        if done:
            self.update(None)
        else:
            statevar = state
            self.update(statevar)




