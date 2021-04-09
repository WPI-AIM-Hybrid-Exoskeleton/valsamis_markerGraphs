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
    minY = [1] * length
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
            markerToeX = toe[i].x
            X.append(markerToeX)
            markerToeY = toe[i].y
            if minY[p] > markerToeY:
                minY[p] = markerToeY
            Y.append(markerToeY)
            markerToeZ = toe[i].z
            Z.append(markerToeZ)
        for y in range(len(Y)):
            Y[y] = Y[y] + abs(minY[p])
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

#getMarker(file)



# frames = {}
# frames["stairA"] = [Point.Point(0, 0, 0),
#                     Point.Point(63, 0, 0),
#                     Point.Point(0, 42, 0),
#                     Point.Point(63, 49, 0)]
#
# frames["stairB"] = [Point.Point(0, 0, 0),
#                     Point.Point(49, 0, 0),
#                     Point.Point(28, 56, 0),
#                     Point.Point(70, 70, 0)]
#
# file = "C:/Users/jjval/Downloads/example_code_TPGMM (3)/iLQR_paper_graphs/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_00/subject_00 walk_00.csv"
#
# def getMarkerZ(file, hills, nb_states, name):
#     for i in file:
#         trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=i)
#         marker = trial.vicon.get_markers()
#         toe = marker.get_marker("LTOE")
#         print(marker)
#         type1 = type(marker)
#         print(type1)
#         markers = trial.vicon.get_markers()
#         markers.smart_sort()
#         markers.auto_make_transform(frames)
#         stair = marker.get_frame("stairA")
#         joint = []
#         for t in toe:
#             joint.append(t.z)
#         arr = [joint[h[0]] for h in hill]
#         paths.append(np.array(arr))
#
#     trainer = GMMTrainer.GMMTrainer(paths, name, nb_states, 0.01)
#     trainer.train()
#     runner = GMMRunner.GMMRunner(name + ".pickle")
#     return runner
#
# def getMarkersY(files, hills, nb_states, name):
#
#     paths = []
#
#     for hill, file in zip(hills, files):
#
#         trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
#         marker = trial.vicon.get_markers()
#         markers = trial.vicon.get_markers()
#         markers.smart_sort()
#         markers.auto_make_transform(frames)
#         toe = marker.get_marker("LTOE")
#         stair = marker.get_frame("stairA")
#         joint = []
#         for t in toe:
#             joint.append(abs(t.y ))
#         arr = [joint[h[0]] for h in hill]
#         paths.append(np.array(arr))
#
#     trainer = GMMTrainer.GMMTrainer(paths, name, nb_states, 0.01)
#     trainer.train()
#     runner = GMMRunner.GMMRunner(name + ".pickle")
#     return runner
#
# def get_index(files):
#
#     paths = []
#     for file in files:
#         trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
#         markers = trial.vicon.get_markers()
#         markers.smart_sort()
#         markers.auto_make_transform(frames)
#         hills = trial.get_stairs("LTOE", "stairA")
#         paths.append(hills[0])
#
#     return paths
#
# def get_traj(files, side):
#
#     frames = {}
#
#     frames["stairA"] = [Point.Point(0, 0, 0),
#                         Point.Point(63, 0, 0),
#                         Point.Point(0, 42, 0),
#                         Point.Point(63, 49, 0)]
#
#     frames["stairB"] = [Point.Point(0, 0, 0),
#                         Point.Point(49, 0, 0),
#                         Point.Point(28, 56, 0),
#                         Point.Point(70, 70, 0)]
#
#     paths = []
#     for s, file in zip(side, files):
#         trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
#         markers = trial.vicon.get_markers()
#         markers.smart_sort()
#         markers.auto_make_transform(frames)
#         stair = markers.get_frame("stairA")
#         if s:
#             hills = trial.get_stairs("RTOE", "stairA")
#             toe = markers.get_marker("RTOE")
#         else:
#             hills = trial.get_stairs("LTOE", "stairA")
#             toe = markers.get_marker("LTOE")
#         Y = []
#         Z = []
#         for i in range(len(hills[0])):
#             index = hills[0][i][0]
#             Y.append(toe[index].y )
#             Z.append(toe[index].z)
#
#         #paths.append(hills[0])
#         paths.append((Y,Z))
#     return paths
#
#
# hills = get_index(file)
# nb_states = 10
#
# paths = get_traj(file, [True])
# runner_toeZ1 = getMarkerZ(files,  hills, nb_states, "toeZ")
# runner_toeY1 = getMarkerY(files, hills, nb_states, "toeY")
#
# runner_toeX2 = GMMRunner.GMMRunner("toeY_all" + ".pickle")
# runner_toeZ2 = GMMRunner.GMMRunner("toeZ_all" + ".pickle")
#
# pathY2= runner_toeX2.run()
# pathZ2 = runner_toeZ2.run()
#
# pathY1 = runner_toeX1.run()
# pathZ1 = runner_toeZ1.run()
#
# z = signal.resample(Z[0], len(pathZ2))
# y = signal.resample(Y[0], len(pathY2))
#
# z_fixed = np.convolve(z, np.ones((N,))/N, mode='valid')
# y_fixed = np.convolve(y, np.ones((N,))/N, mode='valid')
#
# ax[0].plot( y_fixed )
# ax[1].plot( z_fixed )

# fileList = ["C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_00/subject_00 walk_00.csv",
#                                "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_01/subject_01_walk_00.csv",
#                                "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_02/subject_02_walk_00.csv",
#                                "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_03/subject_03_walk_00.csv",
#                                "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_04/subject_04_walk_00.csv",
#                                "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_05/subject_05_walk_00.csv",
#                                "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_06/subject_06 walk_00.csv",
#                                "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_07/subject_07 walk_00.csv",
#                                "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_08/subject_08_walking_01.csv",
#                                "C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_10/subject_10 walk_02.csv"]
#
#
#
# get_data()
#
# gen_traj()
#
# get_gmm("C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_00/subject_00 walk_00.csv")
#
# train_model("C:/Users/jjval/OneDrive/Documents/Junior Year/ExoSkeleton/AIM_GaitData-master/Gaiting_stairs/subject_00/subject_00 walk_00.csv", n_rf=16)
#
# get_bic()

