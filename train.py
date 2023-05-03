import numpy as np
import os,sys
import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import SegDataset
# from AttentionUNet import AttU_Net
# from UNet2plus import NestedUNet
from dqn import DQN
from utils import SegmentationMetric, iou_reward
import matplotlib.pyplot as plt


# model_path = "/home/merrytong/workspace/UNetVersion/AttentionUNet_CFD/focalloss_weights_100.pth"
# model_path = "/home/merrytong/workspace/UNetVersion/UNet2plus_CFD/weights_100.pth"
# model_path = "/home/merrytong/workspace/UNetVersion/UNet2plus_CFD_wo-ds/focalloss_weights_100.pth"
imgpath = "/kaggle/input/deepcrack-results1/deepcrack_results1/img"
maskpath = "/kaggle/input/deepcrack-results1/deepcrack_results1/lab"
# imgpath = "G:\DeepCrack\deepcrack_results2\img"
# maskpath = "G:\DeepCrack\deepcrack_results2\lab"

NUM_ACTIONS = 256
MEMORY_CAPACITY = 10000
NUM_STATE = 448*448

# def train(model_path, imgpath, maskpath):
#     filename = []
#     for name in os.listdir(imgpath):
#         filename.append(name.split(".")[0])

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # model = NestedUNet(1, 3, deep_supervision=True).to(device)
#     model = AttU_Net(3, 1).to(device)

#     model.load_state_dict(torch.load(model_path, map_location=device))

#     dqn = DQN(MEMORY_CAPACITY, NUM_STATE, 16, 0.001, 2000, [448,448], [100,256]).to(device)

#     dataset = SegDataset(imgpath, maskpath, transforms.ToTensor())
#     dataloaders = DataLoader(dataset, batch_size = 1)
#     metric = SegmentationMetric(2)

#     num_epoch = 15
#     n_steps = 1000 # 500 steps per image

#     model.eval()
#     for epoch in range(num_epoch):
#         print("*"*30, epoch+1)

#         for batch_idx, (img, lab) in enumerate(dataloaders) :
#             print("="*20, batch_idx)

#             img = img.to(device)
#             lab = lab.to(device)

#             feature_map = model(img)[-1]
#             # print(feature_map.dtype)
#             # exit()

#             lab = torch.squeeze(lab).detach().cpu().numpy()
#             feature_map = torch.squeeze(feature_map).detach().cpu().numpy()
#             # cv2.imwrite("test.png",feature_map*255)
#             # feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))

#             state = feature_map.copy() * 255

#             for step in range(n_steps):
#                 if epoch == 0 and batch_idx < 10:
#                     action = np.random.randint(100, NUM_ACTIONS)
#                 else:
#                     action = dqn.choose_action(state)

#                 # print("action = ", action)
#                 next_state = state.copy()
#                 next_state[(next_state >= action-5) & (next_state <= action+5)] = 255
#                 # print(state.shape, lab.shape)

#                 reward = iou_reward(metric, state, next_state, lab)

#                 state_save = state.reshape(1,-1).squeeze()
#                 next_state_save = next_state.reshape(1,-1).squeeze()
#                 dqn.store_transition(state_save, action, reward, next_state_save)

#                 if dqn.memory_counter == (MEMORY_CAPACITY):
#                     print('\033[1;31mUpdate Prameters!\033[0m')
#                     dqn.learn()
#                 if dqn.memory_counter > MEMORY_CAPACITY:
#                     dqn.learn()  

#                 state = next_state

#         for batch_idx, (img, lab) in enumerate(dataloaders) :
#             img = img.to(device)
#             lab = lab.to(device)
#             feature_map = model(img)[-1]
#             lab = torch.squeeze(lab).detach().cpu().numpy()
#             feature_map = torch.squeeze(feature_map).detach().cpu().numpy()
#             state = feature_map.copy() * 255

#             metric.addBatch(state/255, lab)
#             iou_pre = metric.IntersectionOverUnion()[1]
#             metric.reset()
#             print(batch_idx, "origin", iou_pre)

#             for step in range(1000):
#                 next_state = state.copy()
#                 action = dqn.choose_action(state/255)
#                 next_state[(next_state >= action-5) & (next_state <= action+5)] = 255
#                 state = next_state

#             metric.addBatch(state/255, lab)
#             iou_pre = metric.IntersectionOverUnion()[1]
#             metric.reset()
#             print(batch_idx, "new", iou_pre)

#             if batch_idx >= 10:
#                 break

#         if (epoch + 1) % 1 == 0:
#             torch.save(dqn.state_dict(), 'dqn_epoch_%d.pth' % (epoch + 1))
#             print('\033[1;31mThe trained model is successfully saved!\033[0m')



def train(imgpath, maskpath):
    filename = []
    for name in os.listdir(imgpath):
        filename.append(name.split(".")[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dqn = DQN(MEMORY_CAPACITY, NUM_STATE, 16, 0.001, 2000, [448,448], [100,256]).to(device)

    dataset = SegDataset(imgpath, maskpath, transforms.ToTensor())
    dataloaders = DataLoader(dataset, batch_size=1)
    metric = SegmentationMetric(2)

    num_epoch = 15
    n_steps = 800  

    for epoch in range(num_epoch):
        print("*"*30, epoch+1)

        for batch_idx, (img, lab) in enumerate(dataloaders):
            print("="*20, batch_idx)

            img = img.to(device)
            lab = lab.to(device)

            feature_map = img  # 使用原始图像作为特征图
            lab = torch.squeeze(lab).detach().cpu().numpy()
            feature_map = torch.squeeze(feature_map).detach().cpu().numpy()

            state = feature_map.copy() * 255

            for step in range(n_steps):
                if epoch == 0 and batch_idx < 10:
                    action = np.random.randint(100, NUM_ACTIONS)
                else:
                    action = dqn.choose_action(state)

                next_state = state.copy()
                next_state[(next_state >= action-5) & (next_state <= action+5)] = 255

                reward = iou_reward(metric, state, next_state, lab)

                state_save = state.reshape(1, -1).squeeze()
                next_state_save = next_state.reshape(1, -1).squeeze()
                dqn.store_transition(state_save, action, reward, next_state_save)

                if dqn.memory_counter == MEMORY_CAPACITY:
                    print('\033[1;31mUpdate Prameters!\033[0m')
                    dqn.learn()
                if dqn.memory_counter > MEMORY_CAPACITY:
                    dqn.learn()  

                state = next_state

        for batch_idx, (img, lab) in enumerate(dataloaders):
            img = img.to(device)
            lab = lab.to(device)
            feature_map = img
            lab = torch.squeeze(lab).detach().cpu().numpy()
            feature_map = torch.squeeze(feature_map).detach().cpu().numpy()
            state = feature_map.copy() * 255

            metric.addBatch(state/255, lab)
            iou_pre = metric.IntersectionOverUnion()[1]
            metric.reset()
            print(batch_idx, "origin", iou_pre)

            for step in range(1000):
                next_state = state.copy()
                action = dqn.choose_action(state/255)
                next_state[(next_state >= action-5) & (next_state <= action+5)] = 255
                state = next_state

            metric.addBatch(state/255, lab)
            iou_pre = metric.IntersectionOverUnion()[1]
            metric.reset()
            print(batch_idx, "new", iou_pre)

            if batch_idx >= 10:
                break

        if (epoch + 1) % 1 == 0:
            torch.save(dqn.state_dict(), 'dqn_epoch_%d.pth' % (epoch + 1))
            print('\033[1;31mThe trained model is successfully saved!\033[0m')





# def val(model_path, imgpath, maskpath):
#     filename = []
#     for name in os.listdir(imgpath):
#         filename.append(name.split(".")[0])

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # model = AttU_Net(3, 1).to(device)
#     model = NestedUNet(1, 3, deep_supervision=False).to(device)

#     model.load_state_dict(torch.load(model_path, map_location=device))

#     dqn = DQN(MEMORY_CAPACITY, NUM_STATE, 16, 0.001, 2000, [448,448], [100,256]).to(device)
#     dqn.load_state_dict(torch.load("NestedUNet_wo-ds/dqn_epoch_15.pth", map_location=device))

#     dataset = SegDataset(imgpath, maskpath, transforms.ToTensor())
#     dataloaders = DataLoader(dataset, batch_size = 1)
#     metric = SegmentationMetric(2)

#     model.eval()
#     for batch_idx, (img, lab) in enumerate(dataloaders) :
#         img = img.to(device)
#         lab = lab.to(device)
#         feature_map = model(img)[-1]
#         lab = torch.squeeze(lab).detach().cpu().numpy()
#         feature_map = torch.squeeze(feature_map).detach().cpu().numpy()
#         state = feature_map.copy() * 255

#         metric.addBatch(state/255, lab)
#         iou_pre = metric.IntersectionOverUnion()[1]
#         metric.reset()
#         print(batch_idx, "origin", iou_pre)

#         for step in range(1000):
#             next_state = state.copy()
#             action = dqn.choose_action(state/255)
#             next_state[(next_state >= action-5) & (next_state <= action+5)] = 255
#             state = next_state

#         metric.addBatch(state/255, lab)
#         iou_pre = metric.IntersectionOverUnion()[1]
#         metric.reset()
#         print(batch_idx, "new", iou_pre)

#     # print(metric.confusionMatrix)



def val(imgpath, maskpath):
    filename = []
    for name in os.listdir(imgpath):
        filename.append(name.split(".")[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = AttU_Net(3, 1).to(device)
    # model = NestedUNet(1, 3, deep_supervision=False).to(device)

    # model.load_state_dict(torch.load(model_path, map_location=device))

    dqn = DQN(MEMORY_CAPACITY, NUM_STATE, 16, 0.001, 2000, [448,448], [100,256]).to(device)
    dqn.load_state_dict(torch.load("NestedUNet_wo-ds/dqn_epoch_15.pth", map_location=device))

    dataset = SegDataset(imgpath, maskpath, transforms.ToTensor())
    dataloaders = DataLoader(dataset, batch_size=1)
    metric = SegmentationMetric(2)

    iou_values = []  # 用于保存IoU值的列表

    # model.eval()
    for batch_idx, (img, lab) in enumerate(dataloaders):
        img = img.to(device)
        lab = lab.to(device)
        feature_map = img
        lab = torch.squeeze(lab).detach().cpu().numpy()
        feature_map = torch.squeeze(feature_map).detach().cpu().numpy()
        state = feature_map.copy() * 255

        metric.addBatch(state / 255, lab)
        iou_pre = metric.IntersectionOverUnion()[1]
        metric.reset()
        print(batch_idx, "origin", iou_pre)

        for step in range(1000):
            next_state = state.copy()
            action = dqn.choose_action(state / 255)
            next_state[(next_state >= action - 5) & (next_state <= action + 5)] = 255
            state = next_state

        metric.addBatch(state / 255, lab)
        iou_pre = metric.IntersectionOverUnion()[1]
        metric.reset()
        print(batch_idx, "new", iou_pre)

        iou_values.append(iou_pre)  # 将IoU值添加到列表中

    # 生成指标变化图
    plt.plot(range(len(iou_values)), iou_values)
    plt.xlabel('Batch Index')
    plt.ylabel('IoU')
    plt.title('IoU Variation')
    plt.savefig('iou_variation.png')
    plt.show()

    # print(metric.confusionMatrix)



'''
AttentionUNet
without dqn
[[7474189.   31149.]
 [  48401.   73013.]]

with dqn
[[7468936.   36402.]
 [  43608.   77806.]]


UNet2plus
without dqn
[[7468794.   36544.]
 [  41979.   79435.]]

with dqn
[[7468687.   36651.]
 [  41876.   79538.]]


NestedUNet---failure
without dqn
[[7475441.   29897.]
 [  49969.   71445.]]

with dqn
[[7467448.   37890.]
 [  42445.   78969.]]
'''


def test(model_path, imgpath, maskpath):
    filename = []
    for name in os.listdir(imgpath):
        filename.append(name.split(".")[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = AttU_Net(3, 1).to(device)
    model = NestedUNet(1, 3, deep_supervision=False).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    dqn = DQN(MEMORY_CAPACITY, NUM_STATE, 16, 0.001, 2000, [448,448], [100,256]).to(device)
    dqn.load_state_dict(torch.load("NestedUNet_wo-ds/dqn_epoch_15.pth", map_location=device))

    dataset = SegDataset(imgpath, maskpath, transforms.ToTensor())
    dataloaders = DataLoader(dataset, batch_size = 1)
    metric = SegmentationMetric(2)

    model.eval()
    for batch_idx, (img, lab) in enumerate(dataloaders) :
        img = img.to(device)
        lab = lab.to(device)
        feature_map = model(img)
        lab = torch.squeeze(lab).detach().cpu().numpy()
        feature_map = torch.squeeze(feature_map).detach().cpu().numpy()
        state = feature_map.copy() * 255


        for step in range(1000):
            next_state = state.copy()
            action = dqn.choose_action(state/255)
            next_state[(next_state >= action-5) & (next_state <= action+5)] = 255
            state = next_state

        cv2.imwrite(os.path.join("predicted_NestedUNet_wo-ds_drl/", str(batch_idx)+".png"), state)

train(imgpath, maskpath)
val(imgpath, maskpath)
# test(model_path, imgpath, maskpath)
