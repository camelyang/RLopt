import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from resnet18 import resnet18


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        print ('Building Encoder')
        
        self.resnet = resnet18(pretrained=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 156)

    def forward(self, x):
        features = self.resnet(x)
        avg_features = self.avg_pool(features)
        avg_features = avg_features.view(-1,512)
        action_prob = F.softmax(self.fc1(avg_features))
        return action_prob


class DQN(nn.Module):
    def __init__(self, memory_capacity, num_state, batch_size, lr, num_iteration, img_size, action_space) -> None:
        super(DQN, self).__init__()

        self.MEMORY_CAPACITY = memory_capacity
        self.LR = lr
        self.ACTION_SPACE = action_space
        self.NUM_STATES = num_state
        self.Q_NETWORK_ITERATION = num_iteration
        self.BATCH_SIZE = batch_size
        self.HEIGHT, self.WIDTH = img_size[0],img_size[1]
        self.EPISILON = 0.8   
        self.GAMMA = 0.9

        self.eval_net, self.target_net = Net().cuda(), Net().cuda()
        for name, p in self.eval_net.named_parameters():
            print(name)
        for name, p in self.target_net.named_parameters():
            p.requires_grad = False

        self.learn_step_counter = 0
        self.memory_counter = 0
        # self.memory = np.zeros((self.MEMORY_CAPACITY, self.NUM_STATES * 2 + 2), dtype = np.float32)
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.NUM_STATES + 2), dtype = np.float32)

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()


    def choose_action(self, state):
        # greedy policy
        if np.random.rand() <= self.EPISILON:
            state = torch.from_numpy(state.astype(np.float32)/255).unsqueeze(0).unsqueeze(0).cuda()
            action_value = self.eval_net(state)
            action = torch.max(action_value, 1)[1].item()
            action += 100
        # random policy
        else: 
            action = np.random.randint(self.ACTION_SPACE[0], self.ACTION_SPACE[1])
        return action


    def store_transition(self, state, action, reward, next_state):

        # transition = np.hstack((state, [action, reward]), next_state)
        transition = np.hstack((state, [action, reward]))

        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        # target_net ---> eval_net
        if self.learn_step_counter != 0 and self.learn_step_counter % self.Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\033[1;31mUpdate target net parameters!\033[0m')

        self.learn_step_counter+=1

        # sample batch from memory
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE, replace = False)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.cuda.FloatTensor(batch_memory[:, :self.NUM_STATES])
        batch_action = torch.cuda.LongTensor(batch_memory[:, self.NUM_STATES:self.NUM_STATES+1].astype(int) - 100)
        batch_reward = torch.cuda.FloatTensor(batch_memory[:, self.NUM_STATES+1:self.NUM_STATES+2])
        # batch_next_state = torch.cuda.FloatTensor(batch_memory[:, -self.NUM_STATES:])

        # q_eval
        batch_state = batch_state.view(-1, self.HEIGHT, self.WIDTH)
        batch_next_state = batch_state.clone()
        # print(batch_next_state.shape, batch_action.shape)
        for i in range(batch_next_state.shape[0]):
            batch_next_state[i][(batch_next_state[i] >= batch_action[i]+100-5) & (batch_next_state[i] <= batch_action[i]+100+5)] = 255

        batch_state = batch_state.unsqueeze(1)/255
        batch_next_state = batch_next_state.unsqueeze(1)/255

        q_eval = self.eval_net(batch_state).gather(1, batch_action)  # gather index LongTensor
        q_next = self.target_net(batch_next_state).detach()
        
        ## Vanilla Q learning
        # q_target = batch_reward + self.opt.GAMMA * q_next.max(1)[0].view(self.opt.BATCH_SIZE, 1)

        ## Double Q learning
        q_double = self.eval_net(batch_next_state).detach()
        eval_act = q_double.max(1)[1]
        q_target = batch_reward + self.GAMMA * q_next[:,eval_act].diag().view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        #increase epsilon
        # if self.learn_step_counter % 2100 == 0:
        #     self.EPISILON = self.EPISILON + self.EPISILON_increment if self.EPISILON < self.EPISILON_MAX else self.EPISILON_MAX
        #     print('EPSILON = ' + str(self.EPISILON))
        if self.learn_step_counter % 8000 == 0:
            q = q_eval[:10,:]
            print('q_eval = ' + str(q))
            print("loss = ", loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def adjust_learning_rate(self, lr, epoch, loss_decay):
        lr_new = lr
        if epoch % loss_decay == 0:
            lr_new = lr * 0.1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_new
        if lr_new != lr:
            print('\033[1;31mlearning rate has updated!\033[0m' + '     learning rate = ' + str(lr_new))
        
        return lr_new        