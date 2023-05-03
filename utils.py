import numpy as np
# import tensorflow as tf
import cv2
import os
import json

__all__ = ['SegmentationMetric']

"""
# 注意：此处竖着代表预测值，横着代表真实值
confusionMetric  
L\P     P    N
P      TP    FN
N      FP    TN
"""
class SegmentationMetric(object):
   
    def __init__(self, numClass):
        
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # precision = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return classAcc # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率
 
    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        # np.nanmean 求平均值，nan表示遇到Nan类型，忽略nan
        # 如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89
        meanAcc = np.nanmean(classAcc) 
        return meanAcc 

    def classRecall(self):
        # recall = (TP) / TP + FN
        classRecall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classRecall

    def meanRecall(self):
        classR = self.classRecall()
        meanR = np.nanmean(classR)
        return meanR

    def IntersectionOverUnion(self):
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix) # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表 
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU 

    def meanIntersectionOverUnion(self):

        IoU = self.IntersectionOverUnion()
        mIoU = np.nanmean(IoU) # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel): # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
 
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU = [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape

        imgPredict = (imgPredict >= 0.49).astype(np.int32)
        imgLabel = imgLabel.astype(np.int32)

        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def iou_reward(metric, state, next_state, lab):

    metric.addBatch(state/255, lab)
    iou_score_pre = metric.IntersectionOverUnion()[1]
    metric.reset()
    metric.addBatch(next_state/255, lab)
    iou_score = metric.IntersectionOverUnion()[1]
    metric.reset()

    if iou_score - iou_score_pre > 0:
        reward = 1
    elif iou_score - iou_score_pre == 0:
        reward = 0
    else:
        reward = -1    

    return reward



if __name__ == '__main__':

    label_path = r"D:\UNetPlus\CrackDataset\crackDataset\CFD_masks_test_fix"
    predict_path = r"D:\Python-64\RealRoadCrack\Predict_DAUNet_encoder_100Epoch_CFD_aug_ace"
    # predict_path = r"D:\Python-64\UNet_Version\Predicted_UNet2plus_CFD_aug_ace_100_focalloss"

    # label_path = r"D:\UNetPlus\CrackDataset\dataset-EdmCrack600\test_lab"
    # predict_path = r"D:\Python-64\RealRoadCrack\Predict_MultiResUNet_100Epoch_Edmcrack600_paste"
    # predict_path = r"D:\Python-64\UNet_Version\Predicted_AttentionUNet_edmcrack600_100_paste"

    data_metrics = {}
    metric = SegmentationMetric(2)

    for filename in os.listdir(label_path):
        imgLabel = cv2.imread(os.path.join(label_path, filename),0)/255
        filename = filename.split(".")[0] + ".png"
        imgPredict = cv2.imread(os.path.join(predict_path, filename),0)/255

        imgPredict = (imgPredict > 0.49).astype(np.int64)
        imgLabel = imgLabel.astype(np.int64)

        key = filename
        temp = {}
        
        metric.addBatch(imgPredict, imgLabel)

        pa = metric.pixelAccuracy()
        temp["accuracy"] = pa
        cpa = metric.classPixelAccuracy()
        temp["precision"] = cpa.tolist()
        recall = metric.classRecall()
        temp["recall"] = recall.tolist()

        mpa = metric.meanPixelAccuracy()
        temp["mean precison"] = mpa
        mr = metric.meanRecall()
        temp["mean recall"] = mr

        IoU = metric.IntersectionOverUnion()
        temp["IoU"] = IoU.tolist()
        mIoU = metric.meanIntersectionOverUnion()
        temp["mean IoU"] = mIoU

        data_metrics[filename] = temp

    json_str = json.dumps(data_metrics)
    with open('test_data_DAUNet_encoder.json', 'w') as json_file:
        json_file.write(json_str)