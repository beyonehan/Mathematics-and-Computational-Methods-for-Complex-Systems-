import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import matplotlib.cm as cm
# x=np.linspace(-1,1,3)
# y=np.linspace(-1,1,5)


# print(x)
# print(xx.shape)
# print(xx)
#
# print(y)
# print(yy.shape)
# print(yy)

# Now we use finer grid (larger arrays) to continue our example:

x=np.linspace(-2.0,2.0,100)
y=np.linspace(-2.0,2.0,100)

import matplotlib.pyplot as plt

X, Y = np.meshgrid(x, y)#生成坐标矩阵# y=np.linspace(-2.0,2.0,2000)
xy= np.vstack([X.ravel(),Y.ravel()]).T#xy就是生成的网格，它是遍布在整个画布上的密集的点
# # x1 = np.array([0.00746603 ,-2.85166464e-04,1.46175915e-03])
# # y1 = np.array([1.5853544,1.58451086e+00,1.58551514e+00])
# x1 = np.array([-2.85166464e-04])
# y1 = np.array([1.58451086e+00])

# x1 = np.array([2.991884582163529,1.3065767374435224,-0.4427042160705914,1.1062785684138168,0.0014617591496162157])
# y1 = np.array([0.815967021383194,0.0362656986673977,-0.6025523304734443,2.9286307259224276,1.5855151405845056])
# x1x1,y1y1=np.meshgrid(x1,y1)


def StopCondition(NewHeight, OldHeight):

    if(abs(NewHeight - OldHeight)) < 0.0001:
        return True
    else:
        return False

def GradAscent(StartPt, NumSteps, LRate):
    PauseFlag = 0

    maxmum = 4
    outPutList = []
    for NumStep in range(NumSteps):
        # TO DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape
        # hanli
        height = SimpleLandscape(StartPt[0], StartPt[1])
        # print("GradAscent i = {},startPt = {} height = {}".format(NumStep,StartPt,height))

        # hanli
        # TO DO: Plot point on the landscape
        # Use a markersize that you can see well enough (e.g., * in size 10)

        plt.plot(StartPt[0], StartPt[1],height, 'or', markersize= 10)  # o for disk and r for red
        # plt.show()

        # TO DO: Calculate the gradient at StartPt using SimpleLandscapeGrad or ComplexLandscapeGrad
        gradient = SimpleLandscapeGrad(StartPt[0], StartPt[1])


        # print("GradAscent gradient = {}".format(gradient))
        # TO DO: Calculate the new point and update StartPt
        StartPt = StartPt + LRate * gradient

        NewHeight = SimpleLandscape(StartPt[0],StartPt[1])

        # print("NumStep = {}",NumStep)
        # outPutList.append({'i': NumStep, 'height': height,'NewHeight':NewHeight,'LRate':LRate,'startPoint': StartPt, 'GradAscent': gradient})
   
        if StopCondition(NewHeight,maxmum) or NumStep == NumSteps -1 :
            # currentTimeStamp = time.time()
            # csv_name = str(currentTimeStamp) + '_modeData.csv'
            # df_again = pd.DataFrame(outPutList, columns=['i', 'height','NewHeight','LRate','startPoint','GradAscent'])
            # df_again.to_csv(csv_name, index=False)
            # print("NewHeight = {},OldeHeight = {},distance = {},NumStep = {}".format(NewHeight,height,abs((NewHeight -height)),NumStep))
            return NumStep

        # Ensure StartPt is within the specified bounds (un/comment relevant lines)
        #  StartPt = np.maximum(StartPt, [-2, -2])
        # StartPt = np.minimum(StartPt,[2,2])
        # StartPt = np.maximum(StartPt,[-3,-3])
        # StartPt = np.minimum(StartPt,[7,7])
        # # Pause to view output
        # if PauseFlag:
        #     x = plt.waitforbuttonpress()


# Definition of Simple landscape
def SimpleLandscape(x, y):
    return np.where(1 - np.abs(2 * x) > 0, 1 - np.abs(2 * x) + x + y, x + y)
# compute function z using the grids constructed by xx and yy
# z=np.exp(-np.sin(2*xx)**2-np.cos(2*yy)**2)-0.5

def SimpleLandscapeGrad(x, y):
    g = np.zeros(2)
    if 1 - np.abs(2 * x) > 0:
        if x < 0:
            g[0] = 3
        elif x == 0:
            g[0] = 0
        else:
            g[0] = -1
    else:
        g[0] = 1
    g[1] = 1
    return g



def gradient_simple_findMAXPoints(startPointArr):
     NumStepList = []
     outPutList = []
     for startPoint in startPointArr:
         NumStep = GradAscent(startPoint,NumSteps= 50,LRate= 0.1)

         height = SimpleLandscape(startPoint[0],startPoint[1])

         outPutList.append({'NumStep': NumStep, 'startX': startPoint[0],'startY':startPoint[1],'height':height,'NumSteps': 50,'LRate':0.1})
         print("getNumStepArr for NumStep = {} startPoint = {} height ={} NumSteps= 50,LRate = 0.1 ".format(NumStep,startPoint,height))
         NumStepList.append(NumStep)


     return NumStepList,outPutList

def Hill_climb_findMaxPoints(startPointArr,Numsteps,MaxMutate):
     NumStepList = []
     outPutList = []
     for startPoint in startPointArr:
         NumStep = HillClimb(startPoint,NumSteps,MaxMutate)

         outPutList.append({'NumStep': NumStep, 'startX': startPoint[0],'startY':startPoint[1],'height':height,'NumSteps': Numsteps,'MaxMutate':MaxMutate})
         print("getNumStepArr for NumStep = {} startPoint = {} height ={} NumSteps= {},MaxMutate = {} ".format(NumStep,startPoint,height,Numsteps,MaxMutate))
         NumStepList.append(NumStep)


# Function implementing hill climbing
# Function implementing hill climbing
def HillClimb(StartPt, NumSteps, MaxMutate):
    PauseFlag = 0
    for i in range(NumSteps):
        print("HillClimb i = ",i )
        # TO DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape

        start_height = SimpleLandscape(StartPt[0],StartPt[1])

        # TO DO: Plot point on the landscape
        # Use a markersize that you can see well enough (e.g., * in size 10)
        ax.plot(StartPt[0], StartPt[1] ,start_height,'or', markersize= 10)  # o for disk and r for red
        plt.show()

        # Mutate StartPt into NewPt
        NewPt = Mutate(np.copy(StartPt),
                       MaxMutate)  # Use copy because Python passes variables by references (see Mutate function)

        # Ensure NewPt is within the specified bounds (un/comment relevant lines)
        # NewPt = np.maximum(NewPt, [-2, -2])
        # NewPt = np.minimum(NewPt, [2, 2])
        NewPt = np.maximum(NewPt,[-3,-3])
        NewPt = np.minimum(NewPt,[7,7])

        # TO DO: Calculate the height of the new point

        new_height = SimpleLandscape(NewPt[0],NewPt[1])

        # TO DO: Decide whether to update StartPt or not

        print("New height = {} NewStartPoint ={} ".format(new_height, NewPt))

        if new_height > start_height:
            StartPt = NewPt

            start_height = new_height
#         # Pause to view output
        if PauseFlag:
            x = plt.waitforbuttonpress()

startPointList = xy
NumStepList,outPutList= gradient_simple_findMAXPoints(startPointList)
currentTimeStamp = time.time()
csv_name = str(currentTimeStamp) +'gradient_simple_findMAXPoint.csv'
df_again = pd.DataFrame(outPutList, columns=['NumStep', 'startX','startY','height','GradAscent','NumSteps','LRate'])

df_again.to_csv(csv_name, index=False)


NumStepList,outPutList= gradient_simple_findMAXPoints(startPointList)

# arr = getNumpeStepArr(startPt)
# print("startPointList.count = {},arr.count = {},arr ={}".format(len(startPointList),len(arr),arr) )

# nonePoint = np.array([3.8421052631578947, 3.8421052631578947])
#
# print(GradAscent(nonePoint,NumSteps=50,LRate= 0.1))


# z = ComplexLandscape(X,Y)
#
# z1 = ComplexLandscape(x1x1,y1y1)
# print(z.shape)
# z = SimpleLandscape(xx,yy)

# CS = plt.contour(x, y, z,cmap=cm.rainbow)
#
# CS = plt.pcolormesh(X, Y, z, cmap=cm.jet,shading='flat',edgecolors='yellow')

# plt.scatter(x1x1, y1y1, c=z1,cmap= 'viridis', marker='*', edgecolors='yellow',linewidths=3.0)

# plt.scatter(x1x1, y1y1, z1)

# plt.colorbar(CS,orientation='vertical')
# #
# # plt.clabel(CS, inline=1, fontsize=5)
#
# # plt.title(r'z=exp(-sin(2x)$^2$-cos(2y)$^2$)-0.5')
# # plt.title(r'4 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 15 * (x / 5 - x ** 3 - y ** 5) * exp(\
# #         -x ** 2 - y ** 2) - (1. / 3) * np.exp(-(x + 1) ** 2 - y ** 2) - 1 * (\
# #                      2 * (x - 3) ** 7 - 0.3 * (y - 4) ** 5 + (y - 3) ** 9) * np.exp(-(x - 3) ** 2 - (y - 3) ** 2)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()