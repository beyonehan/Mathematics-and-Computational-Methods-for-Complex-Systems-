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
# x=np.linspace(-1,1,25)
# y=np.linspace(-1,1,25)
amoumt = 10
x=np.linspace(-3.0,7.0,amoumt)
y=np.linspace(-3.0,7.0,amoumt)

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
    outPutList = []
    for NumStep in range(NumSteps):
        # TO DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape
        # hanli
        height = ComplexLandscape(StartPt[0], StartPt[1])
        # print("GradAscent i = {},startPt = {} height = {}".format(NumStep,StartPt,height))

        # hanli
        # TO DO: Plot point on the landscape
        # Use a markersize that you can see well enough (e.g., * in size 10)

        plt.plot(StartPt[0], StartPt[1],height, 'or', markersize= 10)  # o for disk and r for red
        # plt.show()

        # TO DO: Calculate the gradient at StartPt using SimpleLandscapeGrad or ComplexLandscapeGrad
        gradient = ComplexLandscapeGrad(StartPt[0], StartPt[1])



        # print("GradAscent gradient = {}".format(gradient))
        # TO DO: Calculate the new point and update StartPt
        StartPt = StartPt + LRate * gradient

        NewHeight = ComplexLandscape(StartPt[0],StartPt[1])

        # print("NumStep = {}",NumStep)
        # outPutList.append({'i': NumStep, 'height': height,'NewHeight':NewHeight,'LRate':LRate,'startPoint': StartPt, 'GradAscent': gradient})

        if StopCondition(NewHeight,height) or NumStep == NumSteps -1 :
            # currentTimeStamp = time.time()
            # csv_name = str(currentTimeStamp) + '_modeData.csv'
            # df_again = pd.DataFrame(outPutList, columns=['i', 'height','NewHeight','LRate','startPoint','GradAscent'])
            # df_again.to_csv(csv_name, index=False)
            # print("NewHeight = {},OldeHeight = {},distance = {},NumStep = {}".format(NewHeight,height,abs((NewHeight -height)),NumStep))
            return NumStep,NewHeight,gradient

        # Ensure StartPt is within the specified bounds (un/comment relevant lines)
        #  StartPt = np.maximum(StartPt, [-2, -2])
        # StartPt = np.minimum(StartPt,[2,2])
        # StartPt = np.maximum(StartPt,[-3,-3])
        # StartPt = np.minimum(StartPt,[7,7])
        # # Pause to view output
        # if PauseFlag:
        #     x = plt.waitforbuttonpress()

# Function implementing hill climbing
def HillClimb(StartPt, NumSteps, MaxMutate):

    PauseFlag = 0
    for i in range(NumSteps):
        print("HillClimb i = ",i )
        # TO DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape

        start_height = ComplexLandscape(StartPt[0],StartPt[1])

        # TO DO: Plot point on the landscape
        # Use a markersize that you can see well enough (e.g., * in size 10)
        # plt.plot(StartPt[0], StartPt[1] ,start_height,'or', markersize= 10)  # o for disk and r for red
        # plt.show()

        # Mutate StartPt into NewPt
        NewPt = Mutate(np.copy(StartPt), MaxMutate)  # Use copy because Python passes variables by references (see Mutate function)

        # Ensure NewPt is within the specified bounds (un/comment relevant lines)
        # NewPt = np.maximum(NewPt, [-2, -2])
        # NewPt = np.minimum(NewPt, [2, 2])
        NewPt = np.maximum(NewPt,[-3,-3])
        NewPt = np.minimum(NewPt,[7,7])

        # TO DO: Calculate the height of the new point

        new_height = ComplexLandscape(NewPt[0],NewPt[1])

        # TO DO: Decide whether to update StartPt or not

        print(" HillClimb New height = {} NewStartPoint ={} ".format(new_height, NewPt))

        if new_height > start_height:
            StartPt = NewPt

        
        if StopCondition(new_height,start_height) or i  == NumSteps -1 :
            # currentTimeStamp = time.time()
            # csv_name = str(currentTimeStamp) + '_modeData.csv'
            # df_again = pd.DataFrame(outPutList, columns=['i', 'height','NewHeight','LRate','startPoint','GradAscent'])
            # df_again.to_csv(csv_name, index=False)
            # print("NewHeight = {},OldeHeight = {},distance = {},NumStep = {}".format(NewHeight,height,abs((NewHeight -height)),NumStep))
            # return NumStep,NewHeight,gradient
            return i, new_height

        if PauseFlag:
            x = plt.waitforbuttonpress()


def ComplexLandscape(x, y):
    return 4 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 15 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - (1. / 3) * np.exp(-(x + 1) ** 2 - y ** 2) - 1 * (
                       2 * (x - 3) ** 7 - 0.3 * (y - 4) ** 5 + (y - 3) ** 9) * np.exp(-(x - 3) ** 2 - (y - 3) ** 2)


def ComplexLandscapeGrad(x, y):
    g = np.zeros(2)
    g[0] = -8 * np.exp(-(x ** 2) - (y + 1) ** 2) * ((1 - x) + x * (1 - x) ** 2) - 15 * np.exp(-x ** 2 - y ** 2) * (
                (0.2 - 3 * x ** 2) - 2 * x * (x / 5 - x ** 3 - y ** 5)) + (2. / 3) * (x + 1) * np.exp(
        -(x + 1) ** 2 - y ** 2) - 1 * np.exp(-(x - 3) ** 2 - (y - 3) ** 2) * (
                       14 * (x - 3) ** 6 - 2 * (x - 3) * (2 * (x - 3) ** 7 - 0.3 * (y - 4) ** 5 + (y - 3) ** 9))
    g[1] = -8 * (y + 1) * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 15 * np.exp(-x ** 2 - y ** 2) * (
                -5 * y ** 4 - 2 * y * (x / 5 - x ** 3 - y ** 5)) + (2. / 3) * y * np.exp(
        -(x + 1) ** 2 - y ** 2) - 1 * np.exp(-(x - 3) ** 2 - (y - 3) ** 2) * (
                       (-1.5 * (y - 4) ** 4 + 9 * (y - 3) ** 8) - 2 * (y - 3) * (
                           2 * (x - 3) ** 7 - 0.3 * (y - 4) ** 5 + (y - 3) ** 9))
    return g

# TO DO: Mutation function
# Returns a mutated point given the old point and the range of mutation
def Mutate(OldPt, MaxMutate):
    # TO DO: Select a random distance MutDist to mutate in the range (-MaxMutate,MaxMutate)
     MutDist = np.random.uniform(-MaxMutate,MaxMutate)

     index = np.random.randint(0, len(OldPt))
     print('*** OldPt = {},MutDist = {}'.format(OldPt,MutDist))
     OldPt[index] = OldPt[index] + MutDist

     MutatedPt =  OldPt
     print("**** MutatedPt =",MutatedPt)

    # TO DO: Randomly choose which element of OldPt to mutate and mutate by MutDist

     return MutatedPt


def getDescentInfoList(startPointArr):
     NumStepList = []
     outPutList = []
     for startPoint in startPointArr:
         NumStep,height,gradient = GradAscent(startPoint,NumSteps= 50,LRate= 0.35)

         outPutList.append({'NumStep': NumStep, 'startX': startPoint[0],'startY':startPoint[1],'height':height,'NumSteps': 500,'LRate':0.35})
         print("getNumStepArr for NumStep = {} startPoint = {} height ={} NumSteps= 50,LRate = 0.35 ".format(NumStep,startPoint,height))
         NumStepList.append(NumStep)


     return NumStepList,outPutList

def getHillInfoList(startPointArr, stepLimit, maxMutate,LRate):
    
    NumStepList = []
    outPutList = []

    for startPoint in startPointArr:
        NumStep, height = HillClimb(startPoint,stepLimit,maxMutate)
        NumStepList.append(NumStepList)
        outPutList.append({'NumStep': NumStep, 'startX': startPoint[0],'startY':startPoint[1],'height':height,'NumSteps': stepLimit,'LRate':LRate})
    
    return NumStepList,outPutList

# print(type(np.meshgrid))

maxMutate = 1
LRate = 0.35
stepLimit = 50
startPointList = xy
NumStepList,outPutList= getDescentInfoList(startPointList)
currentTimeStamp = time.time()
Descent_csv_name = 'complexGradient-gradient' + 'Lrate =0.35-' + 'NumpSteps = 50'+'-.csv'

def_Descent_data = pd.DataFrame(outPutList )
# df_again = pd.DataFrame(outPutList, columns=['NumStep', 'startX','startY','height','GradAscent','NumSteps','LRate'])

def_Descent_data.to_csv(Descent_csv_name, index=False)

hill__csv_name =  'complexGradient-Hill' + 'Lrate =0.35-' + 'NumpSteps = 50'+'-.csv'

hill_NumStepList,hill_outPutList = getHillInfoList(startPointList,stepLimit, maxMutate,LRate)

def_Hill_Data = pd.DataFrame(hill_outPutList )

def_Hill_Data.to_csv(hill__csv_name, index=False)



# arr = getNumpeStepArr(startPt)
# print("startPointList.count = {},arr.count = {},arr ={}".format(len(startPointList),len(arr),arr) )

# nonePoint = np.array([3.8421052631578947, 3.8421052631578947])
#
# print(GradAscent(nonePoint,NumSteps=50,LRate= 0.035))


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