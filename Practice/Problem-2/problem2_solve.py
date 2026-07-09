import numpy as np
import matplotlib.pyplot as plt

INF=8

def drawDiscreteSignal(signal,title,xlabel,ylabel,saveto=None):
    fig= plt.figure(figsize=(8,5), layout='constrained')
    plt.stem(np.arange(-INF, INF+1),signal)
    plt.xticks(np.arange(-INF, INF+1))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(-5,5)
    plt.yticks(np.arange(-5, 6))
    plt.grid(True, alpha=0.2)
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()

def initializeSignal():
    return np.array([0, 0, 0, 0, 0, 0, 0.5, 2, 1, 0.5, 1, 0, 0, 0, 0, 0, 0])

def time_scale_signal(x,k): #x[n/k]
    new_len=(len(x)-1)*k+1
    result=np.zeros(new_len)
    result[::k]=x
    return result[INF*k-INF:k*INF+INF+1]

# def time_scale_signal_interpolat(x,k):
#     result=time_scale_signal(x,k)
#     for i in range(-INF,INF+1,1):
#         if i%k!=0:
#             left=i-i%k
#             right=left+k
#             left_val=0 if left<-INF else result[left+INF] 
#             right_val= 0 if right>INF else result[right+INF]
#             print(left_val, right_val)
#             result[i+INF]= (left_val + right_val)/2
#             print(i, left, right)
#     return result

##using linear interpolation// However this doesn't fully satisfy the given criteria // so it would be a better choice to use the above implementation
def time_scale_signal_interpolat(x,k):
    new_len=(len(x)-1)*k+1
    result= np.zeros(new_len)
    result[::k]=x
    val_indices=np.arange(0,new_len, k) #the indices of result that are mapped to some other value of x
    all_indices= np.arange(0,new_len,1) #all indices of x
    result=np.interp(all_indices, val_indices, x)
    return result[INF*k-INF:INF*k+INF+1]

    
def main():
    signal=initializeSignal()
    imgRoot="."
    # drawDiscreteSignal(signal,"Original Signal","n", "x[n]", f"{imgRoot}/x[n].jpg")
    drawDiscreteSignal(time_scale_signal(signal,2),"Signal Scale 1","n", "x[n/2]", f"{imgRoot}/x[nby2].jpg")
    drawDiscreteSignal(time_scale_signal_interpolat(signal,2),"Signal Interpolate 1","n", "x[n/2]", f"{imgRoot}/xI[nby2].jpg")
    drawDiscreteSignal(time_scale_signal(signal,3),"Signal Scale 2","n", "x[n/3]", f"{imgRoot}/x[nby3].jpg")
    drawDiscreteSignal(time_scale_signal_interpolat(signal,3),"Signal Interpolate 2","n", "x[n/3]", f"{imgRoot}/xI[nby3].jpg")

main()