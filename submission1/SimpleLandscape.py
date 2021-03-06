from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random


# Definition of Simple landscape
def SimpleLandscape(x, y):
    return np.where(1-np.abs(2*x)>0,1-np.abs(2*x)+x+y,x+y)
    
# Definition of gradient of Simple landscape
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


    # Function to draw a surface (equivalent to ezmesh in Matlab)
# See argument cmap of plot_surface instruction to adjust color map (if so desired)
def DrawSurface(fig, varxrange, varyrange, function):
    """Function to draw a surface given x,y ranges and a function."""
    ax = fig.gca(projection='3d')
    xx, yy = np.meshgrid(varxrange, varyrange, sparse=False)
    z = function(xx, yy)
    ax.plot_surface(xx, yy, z, cmap='RdBu') # color map can be adjusted, or removed! 
    fig.canvas.draw()
    return ax

def StopCondition(NewHeight, OldHeight):

    if(abs(NewHeight - OldHeight)) < 0.0001:
        return True
    else:
        return False
        


    # Function implementing gradient ascent
def GradAscent(StartPt,NumSteps,LRate):
    MaxHeight = 4
    PauseFlag = 0
    for i in range(NumSteps):
        # TO DO: Calculate the 'height' at StartPt using SimpleLandscape or ComplexLandscape
        
        height = SimpleLandscape(StartPt[0],StartPt[1])

        # TO DO: Plot point on the landscape 
        # Use a markersize that you can see well enough (e.g., * in size 10)
        plt.plot(StartPt[0],StartPt[1],height, 'or',markersize=10 )
        plt.show()
        
        # TO DO: Calculate the gradient at StartPt using SimpleLandscapeGrad or ComplexLandscapeGrad
        gradient = SimpleLandscapeGrad(StartPt[0],StartPt[1])
        
        # TO DO: Calculate the new point and update StartPt
        StartPt = StartPt + LRate * gradient
       

        New_height = SimpleLandscape(StartPt[0],StartPt[1])

        print(" GradAscent new_height = {}".format(New_height))
        
        # Ensure StartPt is within the specified bounds (un/comment relevant lines)
        StartPt = np.maximum(StartPt,[-2,-2])
        StartPt = np.minimum(StartPt,[2,2])

        if New_height >= MaxHeight:
            # print('GradAscent 到达极值点 new_height = {} i = {} '.format(New_height,i))
            return True,i 
        elif i == NumSteps -1 :
            # print('GradAscent 没有到达极值点 new_height = {} i = {} '.format(New_height,i))
            return False,i 
        
        # Pause to view output
        if PauseFlag:
            x=plt.waitforbuttonpress()


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

        if New_height >= MaxHeight:
            # print('HillClimb 到达极值点 new_height = {} i = {} '.format(New_height,i))
            return True, i
        elif i == NumSteps - 1:
            # print('HillClimb 没有到达极值点 new_height = {} i = {} '.format(New_height,i))
            return False, i
        #         # Pause to view output
        if PauseFlag:
            x = plt.waitforbuttonpress()



# Template 
# Plot the landscape (un/comment relevant line)
plt.ion()
fig = plt.figure()
ax = DrawSurface(fig, np.arange(-2,2.025,0.025), np.arange(-2,2.025,0.025), SimpleLandscape)
#ax = DrawSurface(fig, np.arange(-3,7.025,0.025), np.arange(-3,7.025,0.025), ComplexLandscape)

# Enter maximum number of iterations of the algorithm, learning rate and mutation range
NumSteps=50
LRate= 0.1;
MaxMutate= 1

# TO DO: choose a random starting point with x and y in the range (-2, 2) for simple landscape, (-3,7) for complex landscape
x = random.uniform(-2,2)
y = random.uniform(-2,2)
StartPt = np.array([x,y])

# Find maximum (un/comment relevant line)
G_reachMaxHeight , G_steps = GradAscent(StartPt,NumSteps,LRate)
H_reachMaxHeight , H_steps = HillClimb(StartPt,NumSteps,MaxMutate)

 
GradAscent_print  =  'GradAscent G_reachMaxHeight = {}, G_steps use  = {},LRate = {} ,NumSteps = {}'.format(G_reachMaxHeight,G_steps,LRate,NumSteps))
print( 'GradAscent G_reachMaxHeight = {}, G_steps use  = {},LRate = {} ,NumSteps = {}'.format(G_reachMaxHeight,G_steps,LRate,NumSteps))
print('HillClimb   H_reachMaxHeight = {}, H_steps use  = {},LRate= {},NumSteps = {} '.format(H_reachMaxHeight,H_steps,LRate,NumSteps))


# x=np.linspace(-2.0,2.0,100)
# y=np.linspace(-2.0,2.0,100)

# X, Y = np.meshgrid(x, y)#生成坐标矩阵# y=np.linspace(-2.0,2.0,2000)
# xy= np.vstack([X.ravel(),Y.ravel()]).T#xy就是生成的网格，它是遍布在整个画布上的密集的点


# def saveLineSpacePointTestToCSV(xList,yList):


