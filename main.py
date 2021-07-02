import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer, GMMTrainer, TPGMMTrainer_old
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner, TPGMMRunner_old
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from scipy import signal

import numpy as np
from GaitAnaylsisToolkit.Session import ViconGaitingTrial


# from generate_data_set_change import get_joint_angles
# from generate_data_set_change import get_data
#
# from generate_plots import gen_traj
# from generate_plots import plot_gmm
# from generate_plots import get_gmm
# from generate_plots import train_model
# from generate_plots import get_bic

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

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from GaitAnaylsisToolkit.LearningTools.Trainer import TPGMMTrainer, GMMTrainer, TPGMMTrainer_old
from GaitAnaylsisToolkit.LearningTools.Runner import GMMRunner, TPGMMRunner_old
from GaitAnaylsisToolkit.LearningTools.Runner import TPGMMRunner
from scipy import signal
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib
from dtw import dtw
import numpy.polynomial.polynomial as poly

# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))

files = ["11_15_20_nathaniel_walking_01.csv",
         # "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_03.csv",
         #"C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_01.csv"
         # "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/11_15_20_nathaniel_walking_04.csv",  #        1   5    6   7    8   9
         "11_15_20_nathaniel_walking_05.csv",
         "11_15_20_nathaniel_walking_06.csv",
         "11_15_20_nathaniel_walking_07.csv",
         "11_15_20_nathaniel_walking_08.csv",
         "11_15_20_nathaniel_walking_09.csv"]
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
    segmented=[]
    segmentedBracketZ=[]
    segmentedBracketY = []
    length = len(files)
    # minY = [1] * length
    zeroX = 0
    zeroY = 0
    zeroZ = 0
    fig, axs = plt.subplots(2)
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
                if abs(segT[countT] - t) == 10 and len(segT) != 0 and countT > 70:
                    segmentedT.append(segT)
                    segT = []
                    countT = -1

        for T in segmentedT:

            for value in T:

                segX.append(X[value])
                segY.append(Y[value])
                segZ.append(Z[value])
                zeroY = segY[0]

            segmentedX.append(segX)
            segmentedY.append(segY-zeroY)
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
        axs[0].plot(segmentedZ[i])
        axs[1].plot(segmentedY[i])


    segmented.append(segmentedZ)
    segmented.append(segmentedY)
    plt.show()
    return segmented;

arrayCurves = getMarker(files)

def plotfunction(filename):

    runner = TPGMMRunner.TPGMMRunner(filename)

    path = runner.run()

    fig, axs = plt.subplots(2)

    print(path)
    for p in zaxis:
        axs[0].plot(p)
        axs[0].plot(path[:, 0], linewidth=4, color='black')

    for p in yaxis:
        axs[1].plot(p)
        axs[1].plot(path[:, 1], linewidth=4, color='black')
    plt.show()


def PolyTrain(arrayCurves, file_name, bins=15, save=True):
    zaxis = []
    yaxis = []
    for index in range(len(arrayCurves)):
        if index == 0:
            for value in range(len(arrayCurves[index])):
                z_prime = np.array(arrayCurves[index][value])
                zaxis.append(z_prime / 1000)
        if index == 1:
            for value in range(len(arrayCurves[index])):
                y_prime = np.array(arrayCurves[index][value])
                yaxis.append(y_prime / 1000)


    trainer = TPGMMTrainer.TPGMMTrainer(demo=[zaxis, yaxis],
                                        file_name="simpletest",
                                        n_rf=bins,
                                        dt=0.01,
                                        reg=[1e-4],
                                        poly_degree=[10, 10])

    return trainer.train(save)
    # runner = TPGMMRunner.TPGMMRunner("simpletest")
    #
    # path = runner.run()
    #
    # fig, axs = plt.subplots(2)
    #
    # print(path)
    # for p in zaxis:
    #     axs[0].plot(p)
    #     axs[0].plot(path[:, 0], linewidth=4, color='black')
    #
    # for p in yaxis:
    #     axs[1].plot(p)
    #     axs[1].plot(path[:, 1], linewidth=4, color='black')
    #
    # # for x in range(5):
    # #     my_data = runner._data["dtw"][1]
    # #     path1 = my_data[x]["path"][0]
    # #     axs[1].plot(path1)
    # return trainer
    # plt.show()

 #trainer = PolyTrain(arrayCurves)

def plot_gmm(Mu, Sigma, ax=None):
    nbDrawingSeg = 35
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)
    X = []
    nb_state = len(Mu[0])
    patches = []

    for i in range(nb_state):
        w, v = np.linalg.eig(Sigma[i])
        R = np.real(v.dot(np.lib.scimath.sqrt(np.diag(w))))
        x = R.dot(np.array([np.cos(t), np.sin(t)])) + np.matlib.repmat(Mu[:, i].reshape((-1, 1)), 1, nbDrawingSeg)
        x = x.transpose().tolist()
        patches.append(Polygon(x, edgecolor='r'))
        ax.plot(Mu[0, i], Mu[1, i], 'm*', linewidth=10)

    p = PatchCollection(patches, edgecolor='k', color='green', alpha=0.8)
    ax.add_collection(p)

    return p

def get_gmm(trainer):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}

    matplotlib.rc('font', **font)

    nb_states = 10

    runner = TPGMMRunner.TPGMMRunner(trainer)

    fig0, ax = plt.subplots(2,sharex=True)

    sIn = runner.get_sIn()
    tau = runner.get_tau()
    l = runner.get_length()
    motion = runner.get_motion()
    mu = runner.get_mu()
    sigma = runner.get_sigma()
    currF = runner.get_expData()

    # plot the forcing functions
    angles = get_data()
    for i in range(len(angles["Lhip"])): # Hard code number of demonstrations
        ax[0].plot(sIn, tau[1, i * l: (i + 1) * l].tolist(), color="b")
        ax[1].plot(sIn, tau[2, i * l: (i + 1) * l].tolist(), color="b")


    ax[0].plot(sIn, currF[0].tolist(), color="y", linewidth=5)
    ax[1].plot(sIn, currF[1].tolist(), color="y", linewidth=5)


    sigma0 = sigma[:, :3, :2]
    sigma1 = sigma[:, :3, :2]


    sigma1 = np.delete(sigma1, 1, axis=1)
    sigma0 = np.delete(sigma0, 1, axis=1)


    p = plot_gmm(Mu=np.array([mu[0,:], mu[1,:] ]), Sigma=sigma0, ax=ax[0])
    p = plot_gmm(Mu=np.array([mu[0, :], mu[2, :]]), Sigma=sigma1, ax=ax[1])

    fig0.suptitle('Forcing Function')


    ax[0].set_ylabel('F')
    ax[1].set_ylabel('F')


    # fig0.tight_layout(pad=1.0, h_pad=0.15, w_pad=None, rect=None)
    ax[0].set_title("Z Axis")
    ax[1].set_title("Y Axis")


    plt.show()

# get_gmm("simpletest")


def get_BIC(arrayCurves, file_name):

    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}

    matplotlib.rc('font', **font)

    for j in range(1):
        BIC = {}
        for i in range(15,18):
            print(i)
            data = PolyTrain(arrayCurves, file_name+str(i), bins=i, save=True)
            BIC[i] = data["BIC"]
            print(data["BIC"])

        plt.plot(list(BIC.keys()), list(BIC.values()))

    plt.xlabel("Bins")
    plt.ylabel("BIC")
    plt.title("BIC score for Walking")
    plt.show()




def calculate_imitation_metric(file_name):
    angles = get_data()
    demos = [angles["zaxis"], angles["yaxis"]]
    runner = TPGMMRunner.TPGMMRunner(file_name)
    path = runner.run()
    print(path[:,0])


    alpha = 1.0
    manhattan_distance = lambda x, y: abs(x - y)

    costs = []
    for i in range(3):
        imitation = path[:, i]
        T = len(imitation)
        M = len(demos[i])
        metric = 0.0
        t = []
        t.append(1.0)
        for k in range(1, T):
            t.append(t[k - 1] - alpha * t[k - 1] * 0.01)  # Update of decay term (ds/dt=-alpha s) )
        t = np.array(t)

        for m in range(M):
            d, cost_matrix, acc_cost_matrix, path_im = dtw(imitation, demos[i][m], dist=manhattan_distance)
            data_warp = [demos[i][m][path_im[1]][:imitation.shape[0]]]
            coefs = poly.polyfit(t, data_warp[0], 20)
            ffit = poly.Polynomial(coefs)
            y_fit = ffit(t)
            metric += np.sum(abs(y_fit - imitation.flatten()))

        costs.append(metric / (M * T))
        print("cost")
        print(costs)
    return costs

get_BIC(arrayCurves, "simpletest")