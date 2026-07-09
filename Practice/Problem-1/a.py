import numpy as np
import matplotlib.pyplot as plt

INF = 8

def drawDiscreteSignal(signal: np.ndarray,title: str,xlabel: str,ylabel: str,grid: bool,figsize=(8,5), saveto=None):
    fig = plt.figure(figsize=figsize, layout='constrained')
    plt.stem(np.arange(-INF,INF+1,1), signal)
    plt.xticks(np.arange(-INF,INF+1,1))
    y_range=[min(np.min(signal)-1,0), np.max(np.max(signal)+1)]
    plt.ylim(*y_range) #unpack/splat operator for 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid, alpha=0.2)

    if saveto is not None:
        plt.savefig(saveto)

    plt.show() #this show must come after savefig because show always clear the plt once you close the terminal. So if savefig comes after it then you will not be able to see any picture being saved!!


def initiateSignal():
    return np.array([0,0,0,1,0,3,2,1,2,0,0,5,-3,0,0,0,0])

def signalShift(signal: np.ndarray, k: np.int8)-> np.ndarray:
    result=np.zeros_like(signal)
    if k>0:
        result[k:]=signal[:-k] 
    elif k<0:
        k=-k
        result[:-k]=signal[k:]
    else:
        result=np.copy(signal)
    return result

def signalScale(signal: np.ndarray, k:np.int8)->np.ndarray:
    result=np.zeros_like(signal)
    leftHalf=signal[INF::k]
    rightHalf=signal[INF::-k]
    result[INF:INF+np.size(leftHalf)]=leftHalf
    result[INF-np.size(rightHalf):INF]=rightHalf
    return result
def main():
    signal=initiateSignal()
    imageRoot="."
    drawDiscreteSignal(signal=signal,title="Input Signal", xlabel="n", ylabel="x[n]",grid=True, saveto=f"{imageRoot}/x[n].jpg")
    # drawDiscreteSignal(signal=signalShift(signal,3),title="Signal Shift 1", xlabel="n", ylabel="x[n-3]",grid=True, saveto=f"{imageRoot}/x[n-3].jpg")
    # drawDiscreteSignal(signal=signalShift(signal,-3),title="Signal Shift 2", xlabel="n", ylabel="x[n+3]",grid=True, saveto=f"{imageRoot}/x[n+3].jpg")
    # drawDiscreteSignal(signal=signalShift(signal,10),title="Signal Shift 3", xlabel="n", ylabel="x[n-10]",grid=True, saveto=f"{imageRoot}/x[n-10].jpg")
    # drawDiscreteSignal(signal=signalShift(signal,0),title="Signal Shift 4", xlabel="n", ylabel="x[n-0]",grid=True, saveto=f"{imageRoot}/x[n-0].jpg")
    drawDiscreteSignal(signal=signalScale(signal,2),title="Signal Scale 1", xlabel="n", ylabel="x[2n]",grid=True, saveto=f"{imageRoot}/x[2n].jpg")
   

main()
