import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer, GMMTrainer, TPGMMTrainer_old
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner, TPGMMRunner_old
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from scipy import signal

import numpy as np
from GaitAnaylsisToolkit.Session import ViconGaitingTrial


from generate_data_set_change import get_joint_angles
from generate_data_set_change import get_data

from generate_plots import gen_traj
from generate_plots import plot_gmm
from generate_plots import get_gmm
from generate_plots import train_model
from generate_plots import get_bic

from GaitCore.Core import Point
import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Trainer import GMMTrainer
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner
from GaitAnaylsisToolkit.Session import ViconGaitingTrial


import numpy as np

from scipy import signal
from dtw import dtw

import os

# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))

files = ["C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_01.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_03.csv",
                                # "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_02.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_03.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_04.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_05.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_06.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_07.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_08.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_09.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_10.csv"]

#file = "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_00/subject_00 walk_00.csv"

def getMarker(files):
    fig = plt.figure()
    arrX=[]
    arrY=[]
    arrZ=[]
    length = len(files)
    # minY = [1] * length
    zeroX = 0
    zeroY = 0
    zeroZ = 0
    p = 0
    for file in files:
        print(file)
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        marker = trial.vicon.get_markers()
        toe = marker.get_marker("LTOE")
        l = len(toe)
        print(l)
        X=[]
        Y=[]
        Z=[]
        for i in range(l):
            # if toe[0].y > toe[l-1].y:
            #     np.flip(toe)
            markerToeX = toe[i].x
            zeroX = toe[0].x

            X.append(markerToeX)
            # if toe[0].y > toe[l-1].y:
            #     np.flip(toe.y)
            markerToeY = toe[i].y
            zeroY = toe[0].y

            # if toe[0].y > toe[l-1].y:
            #     np.flip(toe.z)
            # if minY[p] > markerToeY:
            #     minY[p] = markerToeY
            Y.append(markerToeY)
            markerToeZ = toe[i].z
            zeroZ = toe[0].z

            Z.append(markerToeZ)
        print(zeroX)
        print((-1*(np.sign(zeroX))))
        print((-1*(np.sign(zeroX)))*X[0])
        print(zeroY)
        print((-1*(np.sign(zeroY))))
        print((-1*(np.sign(zeroY)))*Y[0])
        print(zeroZ)
        print((-1*(np.sign(zeroZ)))*Z[0])
        for y in range(len(X)):
            if(np.sign(zeroX) == 1):
                X[y] = X[y] +(-1*(np.sign(zeroX))) * zeroX
            if (np.sign(zeroX) == -1):
                X[y] = X[y] + (1 * (np.sign(zeroX))) * zeroX
            if (np.sign(X[y]) == -1):
                X[y] = -X[y]
            else:
                X[y] = X[y]
        print(X[0])
        for y in range(len(Y)):
            if (np.sign(zeroY) == 1):
                Y[y] = Y[y] + (-1 * (np.sign(zeroY))) * zeroY
            if (np.sign(zeroY) == -1):
                Y[y] = Y[y] + (1 * (np.sign(zeroY))) * zeroY
            if (np.sign(Y[y]) == -1):
                Y[y] = -Y[y]
            else:
                Y[y] = Y[y]
        print(Y[0])
        for y in range(len(Z)):
            if (np.sign(zeroZ) == 1):
                Z[y] = Z[y] + (-1 * (np.sign(zeroZ))) * zeroZ
            if (np.sign(zeroZ) == -1):
                Z[y] = Z[y] + (1 * (np.sign(zeroZ))) * zeroZ
            if (np.sign(Z[y]) == -1):
                Z[y] = -Z[y]
            else:
                Z[y] = Z[y]

        print(Z[0])
        # for y in range(len(Y)):
        #     Y[y] = Y[y] + abs(minY[p])
        # print(minY)
        # p += 1
        # print(Y[0])

        arrX.append(X)
        arrY.append(Y)
        arrZ.append(Z)

    for i in range(len(arrX)):
        plt.plot(arrX[i])
        plt.plot(arrY[i])
        plt.plot(arrZ[i])






    plt.show()

getMarker(files)