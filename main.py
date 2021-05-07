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

from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer
from random import seed
from random import gauss
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))

files = ["C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_05.csv",
        # "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_03.csv",
                                #"C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_01.csv"
         # "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_04.csv",          1   5    6   7    8   9
         # "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_05.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_06.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_07.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_08.csv",
         "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_09.csv"]
         # # "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_10.csv"]

#file = "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_00/subject_00 walk_00.csv"

seed(1)
count = 200


def getMarker(files):
    fig = plt.figure()
    arrX=[]
    arrY=[]
    arrZ=[]
    segmentedT=[]
    segmentedX=[]
    segmentedY=[]
    segmentedZ=[]
    segmentedBracketZ=[]
    segmentedBracketY = []
    length = len(files)
    # minY = [1] * length
    zeroX = 0
    zeroY = 0
    zeroZ = 0
    for file in files:
        print(file)
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        marker = trial.vicon.get_markers()
        toe = marker.get_marker("RTOE")
        l = len(toe)
        print(l)
        X=[]
        Y=[]
        Z=[]
        segT=[]
        segX=[]
        segY=[]
        segZ=[]
        countT = -1
        for i in range(l):                 #Grabs the marker points and appends them to arrays
            markerToeX = toe[i].x
            zeroX = toe[0].x
            X.append(markerToeX)

            markerToeY = toe[i].y
            zeroY = toe[0].y
            Y.append(markerToeY)

            markerToeZ = toe[i].z
            zeroZ = toe[0].z
            Z.append(markerToeZ)

        for y in range(len(X)):             #Flips the markers if they are reversed and starts them at 0
            if(np.sign(zeroX) == 1):
                X[y] = X[y] +(-1*(np.sign(zeroX))) * zeroX
            if (np.sign(zeroX) == -1):
                X[y] = X[y] + (1 * (np.sign(zeroX))) * zeroX
            if (np.sign(X[y]) == -1):
                X[y] = -X[y]
            else:
                X[y] = X[y]

        for y in range(len(Y)):
            if (np.sign(zeroY) == 1):
                Y[y] = Y[y] + (-1 * (np.sign(zeroY))) * zeroY
            if (np.sign(zeroY) == -1):
                Y[y] = Y[y] + (1 * (np.sign(zeroY))) * zeroY
            if (np.sign(Y[y]) == -1):
                Y[y] = -Y[y]
            else:
                Y[y] = Y[y]

        for y in range(len(Z)):
            if (np.sign(zeroZ) == 1):
                Z[y] = Z[y] + (-1 * (np.sign(zeroZ))) * zeroZ
            if (np.sign(zeroZ) == -1):
                Z[y] = Z[y] + (1 * (np.sign(zeroZ))) * zeroZ
            if (np.sign(Z[y]) == -1):
                Z[y] = -Z[y]
            else:
                Z[y] = Z[y]
#Segment

        for t in range(len(Z)):             #Segments the points
            if t+1 == len(Z):
                break
            if Z[t] > 28:
                countT += 1
                segT.append(t)
            if countT != -1:
                if abs(segT[countT] - t) == 10 and len(segT) != 0:
                    segmentedT.append(segT)
                    segT = []
                    countT = -1

        print(segmentedT)
        print("-------------------------------------------------------")
        print("-------------------------------------------------------")
        print("-------------------------------------------------------")

        for T in segmentedT:

            for value in T:

                segX.append([X[value]])
                segY.append([Y[value]])
                segZ.append(Z[value])

            segmentedX.append(segX)
            segmentedY.append(segY)
            segmentedZ.append(segZ)

            segX = []
            segY = []
            segZ = []

        segmentedT = []


        arrX.append(X)
        arrY.append(Y)
        arrZ.append(Z)

    for i in range(len(segmentedZ)):
        # plt.plot(arrX[i])
        # plt.plot(arrY[i])
        # plt.plot(arrZ[i])
        # plt.plot(segmentedX[i])
        # plt.plot(segmentedY[i])
        # print(segmentedY)
        print(segmentedZ[i])

        plt.plot(segmentedZ[i])

    plt.show()
    return segmentedZ;

arrayCurves = getMarker(files)
print("---------------------------------------")
print(arrayCurves[1])
print("---------------------------------------")


def PolyTrain(arrayCurves,arrayCurves1, arrayCurves2):

    z_prime = np.array(arrayCurves)
    z_prime1 = np.array(arrayCurves1)
    z_prime2 = np.array(arrayCurves2)
    zaxis = []
    zaxis.append(z_prime / 1000)
    zaxis.append(z_prime1 / 1000)
    zaxis.append(z_prime2 / 1000)

    trainer = TPGMMTrainer.TPGMMTrainer(demo=[zaxis, zaxis],
                                        file_name="simpletest",
                                        n_rf=6,
                                        dt=0.01,
                                        reg=[1e-4],
                                        poly_degree=[3, 3])

    trainer.train()
    runner = TPGMMRunner.TPGMMRunner("simpletest")

    path = runner.run()

    fig, axs = plt.subplots(2)

    print(path)
    for p in zaxis:
        axs[0].plot(p)
        axs[0].plot(path[:, 0], linewidth=4, color='black')

    for p in zaxis:
        axs[1].plot(p)
        axs[1].plot(path[:, 1], linewidth=4, color='black')

    # for x in range(5):
    #     my_data = runner._data["dtw"][1]
    #     path1 = my_data[x]["path"][0]
    #     axs[1].plot(path1)

    plt.show()

for index in range(len(arrayCurves)):  # This needs to be in bursts of 3 for each file  It doesnt work right now
    if index % 3 == 0:
        PolyTrain(arrayCurves[index],arrayCurves[index+1],arrayCurves[index+2])

