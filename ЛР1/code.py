import scipy.linalg as sp
import math as m
import numpy as np
from typing import NamedTuple

class Point(NamedTuple):
    x: float
    y: float
    z: float

sigma = 0.1
alpha = 1e-7
A = np.zeros((3, 3), dtype=np.float32)
b = np.zeros(3, dtype=np.float32)
I = np.array([11, 3, 1.1], dtype=np.float32)
IReal = np.array([10, 2, 0.1], dtype=np.float32)
IRegu = I.copy()
Ns = np.array([Point(300, 0, 0), Point(600 ,0 , 0), Point(1100, 0, 0)])
Ms = np.array([Point(200, 0, 0), Point(500 ,0 , 0), Point(1000, 0, 0)])
As = np.array([Point(0, -500, 0), Point(0 ,0 , 0), Point(0, 500, 0)])
Bs = np.array([Point(100, -500, 0), Point(100 ,0 , 0), Point(100, 500, 0)])

def V(recieverNum):
    sum = 0.0
    for i in range(0, 3):
        sum += I[i] / (2.0 * m.pi * sigma) * ((1.0 / Distance(Ms[recieverNum], Bs[i]) - 1.0 / Distance(Ms[recieverNum], As[i])) - (1.0 / Distance(Ns[recieverNum], Bs[i]) - 1.0 / Distance(Ns[recieverNum], As[i])))
    return sum

def VReal(recieverNum):
    sum = 0.0
    for i in range(0, 3):
        sum += IReal[i] / (2.0 * m.pi * sigma) * ((1.0 / Distance(Ms[recieverNum], Bs[i]) - 1.0 / Distance(Ms[recieverNum], As[i])) - (1.0 / Distance(Ns[recieverNum], Bs[i]) - 1.0 / Distance(Ns[recieverNum], As[i])))
    return sum


def F():
    sum = 0.0
    for i in range(0,3):
        w = 1.0 / V(i)
        sum += (w * (V(i) - VReal(i)))**2
    return sum

def VDer(recieverNum, arg):
    return 1.0 / (2.0 * m.pi * sigma) * ((1.0 / Distance(Ms[recieverNum], Bs[arg]) - 1.0 / Distance(Ms[recieverNum], As[arg])) - (1.0 / Distance(Ns[recieverNum], Bs[arg]) - 1.0 / Distance(Ns[recieverNum], As[arg])))

def Distance(a: Point, b: Point):
    return m.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2 + (b[2] - a[2])**2)

# print("Vreal 0  = ", VReal(0))
# print("Vreal 1  = ", VReal(1))
# print("Vreal 2  = ", VReal(2))

# print("VDer 0, I1 = ", VDer(0, 2))
# print("VDer 1, I1 = ", VDer(1, 2))
# print("VDer 2, I1 = ", VDer(2, 2))


iter = 0
while F() >= 1e-14 and iter < 2:
    for q in range(0, 3):
        for s in range(0, 3):
            sum = 0.0
            for i in range(0, 3):
                wi = 1.0 / V(i)
                derQ = VDer(i, q)
                derS = VDer(i, s)
                sum += wi**2 * derQ * derS
            A[q, s] = sum
        A[q, q] += alpha
        
        sumb = 0.0
        for i in range(0, 3):
            wi = 1.0 / V(i)
            derQ = VDer(i, q)
            vDelta = V(i) - VReal(i)
            sumb -= wi**2 * derQ * vDelta
        b[q] = sumb
        b[q] -= alpha * (I[q] - IRegu[q])
    print(b)
    I += sp.solve(A, b, assume_a="sym")
    #print(f"{F():e}")
    #print(I)
    iter += 1
print(I)