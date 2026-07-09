import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Time axis
# ----------------------------
T_MIN, T_MAX, N = -4.0, 4.0, 4001


def x_of_t(t: np.ndarray) -> np.ndarray:
    """
    Base signal x(t): sinusoidal signal
    """
    return (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    )


# ==========================================================
# ANSWER IMPLEMENTATION
# ==========================================================

def interpolate_signal(
    t_original: np.ndarray,
    x_original: np.ndarray,
    k
) -> np.ndarray:
    """
    Interpolate using average of two neighboring samples.
    """
    y=time_scale(t_original,x_original,k)
    for i in range(np.size(y)):
        if i%k!=0:
            leftIndex=i-i%k
            rightIndex=leftIndex+k
            leftVal=0 if leftIndex<0 else y[leftIndex]
            rightVal=0 if rightIndex>=np.size(y) else y[rightIndex]
            y[i]=(leftVal+rightVal)/2
    # raise NotImplementedError
    return y

def time_scale(
    t: np.ndarray,
    x: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Time sub-scaling:
        y(t) = x(t / k)
    """
    newLen=np.size(t)+(np.size(t)-1)*(k-1)
    result=np.zeros(newLen)
    result[::k]=x
    halfLen_t=np.size(t)//2
    halfLen_result=np.size(result)//2
    return result[halfLen_result-halfLen_t:halfLen_result+halfLen_t+1]

def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    """
    Plot graphs.
    """
    plt.plot(t,x)
    plt.plot(t,y)
    # plt.show()
    # raise NotImplementedError


# ----------------------------
# Main
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    k = 3  # sub-scaling factor
    y = interpolate_signal(t, x, k)

    plot_pair(
        t,
        x,
        y,
        title=f"Time Sub-scaling: y(t) = x(t / {k})"
    )
    plt.show()


if __name__ == "__main__":
    main()
