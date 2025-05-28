import numpy as np
import os
import cv2 as cv
import pandas as pd
from numpy import matlib
from COA import COA
from GOA import GOA
from Global_Vars import Global_Vars
from LOA import LOA
from Model_CNN import Model_CNN
from Model_DQN import Model_DQN
from Model_ResDAM import Model_ResDAM
from Model_SVM import Model_SVM
from Model_TRDUnetPlusPlus import Model_TRDUnetPlusPlus
from Objective_Function import objfun_cls
from Proposed import Proposed
from WPA import WPA
from Plot_Results import *
from Image_Results import *

no_of_dataset = 2


def ReadText(filename):
    f = open(filename, "r")
    lines = f.readlines()
    Tar = []
    fileNames = []
    for lineIndex in range(len(lines)):
        if lineIndex and '||' in lines[lineIndex]:
            line = [i.strip() for i in lines[lineIndex].strip().strip('||').replace('||', '|').split('|')]
            fileNames.append(line[0])
            Tar.append(int(line[2]))
    Tar = np.asarray(Tar)
    uniq = np.unique(Tar)
    Target = np.zeros((len(Tar), len(uniq))).astype('int')
    for i in range(len(uniq)):
        index = np.where(Tar == uniq[i])
        Target[index, i] = 1
    return fileNames, Target


def Read_Image(Filename):
    image = cv.imread(Filename)
    image = np.uint8(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (512, 512))
    return image


def Read_Images(Directory):
    Images = []
    out_folder = os.listdir(Directory)
    for i in range(len(out_folder)):
        print(i)
        filename = Directory + out_folder[i]
        image = Read_Image(filename)
        Images.append(image)
    return Images


def Read_Datset_PH2(Directory, fileNames):
    Images = []
    GT = []
    folders = os.listdir(Directory)
    for i in range(len(folders)):
        if folders[i] in fileNames:
            image = Read_Image(Directory + folders[i] + '/' + folders[i] + '_Dermoscopic_Image/' + folders[i] + '.bmp')
            gt = Read_Image(Directory + folders[i] + '/' + folders[i] + '_lesion/' + folders[i] + '_lesion.bmp')
            Images.append(image)
            GT.append(gt)
    return Images, GT


def Read_CSV(Path):
    df = pd.read_csv(Path)
    values = df.to_numpy()
    value = values[:, 6]
    uniq = np.unique(value)
    Target = np.zeros((len(value), len(uniq))).astype('int')
    for i in range(len(uniq)):
        index = np.where(value == uniq[i])
        Target[index, i] = 1
    return Target


# Read Datasets
an = 0
if an == 1:
    Images1 = Read_Images('./Datasets/HAM10000/Images/')
    np.save('Images_1.npy', Images1)

    fileNames, Target2 = ReadText('./Datasets/PH2Dataset/PH2_dataset.txt')
    Images2, GT = Read_Datset_PH2('./Datasets/PH2Dataset/PH2 Dataset images/', fileNames)
    np.save('Images_2.npy', Images2)
    np.save('GT_2.npy', GT)
    np.save('Target_2.npy', Target2)

# GroundTruth for Dataset1
an = 0
if an == 1:
    im = []
    img = np.load('Images_1.npy', allow_pickle=True)
    for i in range(len(img)):
        print(i)
        image = img[i]
        minimum = int(np.min(image))
        maximum = int(np.max(image))
        Sum = ((minimum + maximum) / 2)
        ret, thresh = cv.threshold(image, Sum, 255, cv.THRESH_BINARY_INV)
        im.append(thresh)
    np.save('GT_1.npy', im)

# Generate Target for Dataset 1
an = 0
if an == 1:
    Tar = []
    Ground_Truth = np.load('GT_1.npy', allow_pickle=True)
    for i in range(len(Ground_Truth)):
        image = Ground_Truth[i]
        result = image.astype('uint8')
        uniq = np.unique(result)
        if len(uniq) > 1:
            Tar.append(1)
        else:
            Tar.append(0)
    Tar = (to_categorical(np.asarray(Tar).reshape(-1, 1))).astype('int')
    np.save('Target_1.npy',Tar)


# Segmentation by Transformer-based Residual Dense Unet++ (TRDUnet++)
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        GT = np.load('GT_' + str(n + 1) + '.npy', allow_pickle=True)
        Eval, Segmented_Image = Model_TRDUnetPlusPlus(Images, GT)
        np.save('Method5_Dataset' + str(n + 1) + '.npy', Segmented_Image)

# Optimization for Residual Densenet with Attention Mechanism (AResDAM)
an = 0
if an == 1:
    BestSol = []
    for n in range(no_of_dataset):
        Images = np.load('Method5_Dataset' + str(n + 1) + '.npy', allow_pickle=True)
        Tar = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Images = Images
        Global_Vars.Target = Tar
        Npop = 10
        Chlen = 3  # Here we optimized Hidden Neuron Count, No of epoches, Steps per epoch in
        xmin = matlib.repmat([5, 5, 50], Npop, 1)
        xmax = matlib.repmat([255, 50, 250], Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(Chlen):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = objfun_cls
        Max_iter = 50

        print("CO...")
        [bestfit1, fitness1, bestsol1, time1] = COA(initsol, fname, xmin, xmax, Max_iter)  # CO

        print("GOA...")
        [bestfit2, fitness2, bestsol2, time2] = GOA(initsol, fname, xmin, xmax, Max_iter)  # GOA

        print("WPA...")
        [bestfit3, fitness3, bestsol3, time3] = WPA(initsol, fname, xmin, xmax, Max_iter)  # WPA

        print("LOA...")
        [bestfit4, fitness4, bestsol4, time4] = LOA(initsol, fname, xmin, xmax, Max_iter)  # LOA

        print("Improved LOA...")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Improved LOA

        BestSol.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])  #
    np.save('BestSol.npy', BestSol)


# Classification
an = 0
if an == 1:
    Evaluate_all = []
    for n in range(no_of_dataset):
        Feat = np.load('Method5_Dataset'+str(n+1)+'.npy', allow_pickle=True)
        Tar = np.load('Target_'+str(n+1)+'.npy', allow_pickle=True)
        bests = np.load('BestSol.npy', allow_pickle=True)[n]
        Eval_all = []
        Steps_per_Epoch = [100, 200, 300, 400, 500]
        for m in range(len(Steps_per_Epoch)):  # for all learning percentage
            EVAL = np.zeros((10, 14))
            per = round(len(Feat) * 0.75)
            Train_Data = Feat[:per, :, :]
            Train_Target = Tar[:per, :]
            Test_Data = Feat[per:, :, :]
            Test_Target = Tar[per:, :]
            for j in range(bests.shape[0]):  # for all algorithms
                soln = bests[j]
                EVAL[j, :], pred1 = Model_ResDAM(Train_Data, Train_Target, Test_Data, Test_Target,Steps_per_Epoch[m], soln)  # with Optimization ResDAM
            EVAL[5, :], pred2 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target,Steps_per_Epoch[m])  # CNN Model
            EVAL[6, :], pred3 = Model_SVM(Train_Data, Train_Target, Test_Data, Test_Target,Steps_per_Epoch[m])  # SVM model
            EVAL[7, :], pred4 = Model_DQN(Train_Data, Train_Target, Test_Data, Test_Target,Steps_per_Epoch[m])  # DQN model
            EVAL[8, :], pred5 = Model_ResDAM(Train_Data, Train_Target, Test_Data,Steps_per_Epoch[m],
                                             Test_Target)  # without Optimization ResDAM
            EVAL[9, :] = EVAL[4, :]  # with Optimization ensemble
            Eval_all.append(EVAL)
        Evaluate_all.append(Eval_all)
    np.save('Eval_all.npy', Evaluate_all)


plot_results_stepsperepoch()
plot_results()
plot_Segmentation_results_1()
plotConvResults()
Plot_ROC_Curve()
Image_Results()
Sample_Images()