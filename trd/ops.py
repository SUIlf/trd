import numpy as np
from math import gcd

def my_chop2(sv, eps):
    if eps <= 0.0:
        r = len(sv)
        return r
    sv0 = np.cumsum(abs(sv[::-1]) ** 2)[::-1]
    ff = [i for i in range(len(sv0)) if sv0[i] < eps ** 2]
    if len(ff) == 0:
        return len(sv)
    else:
        return np.amin(ff)

def factor(n):
    """returns all irreducible factors of x in vector
    n : int
    Returns
    -------
    T : vector
        factors
    """
    factors = []
    def get_factor(n):
        x_fixed = 2
        cycle_size = 2
        x = 2
        factor = 1
        while factor == 1:
            for count in range(cycle_size):
                if factor > 1: break
                x = (x * x + 1) % n
                factor = gcd(x - x_fixed, n)
            cycle_size *= 2
            x_fixed = x
        return factor
    while n > 1:
        next = get_factor(n)
        factors.append(next)
        n //= next
    return factors

def cores_initialization(tensor_size, tr_rank):
    tr_cores = []
    ndims = len(tensor_size)
    # print(ndims)
    tr_rank.append(tr_rank[0])
    for n in range(0, ndims):
        tr_cores.append(0.1 * np.random.rand(tr_rank[n], tensor_size[n], tr_rank[n+1]))
    return tr_cores

def lrf_cores_initialization(S, r):
    N = int(np.size(S))
    Z = np.zeros((3,), dtype=np.object)  # 设置 dtype=np.object，可以在矩阵中设置形状不同的子矩阵
    for i in range(N-1):
        Z[i] = np.random.randn(r[i], S[i], r[i+1])
    Z[N-1] = np.random.randn(r[N-1], S[N-1], r[0])
    return Z

def circshift(od):
    item = od.pop(0)
    od.append(item)
    return od

def Z_neq(Z, n):
    Z = np.roll(Z, -n - 1)  # arrange Z{n} to the last core, so we only need to multiply the first N-1 core
    N = np.size(Z, 0)
    P = Z[0]

    for i in range(N - 2):
        zl = np.reshape(P, (int(P.size / (np.size(Z[i], 2))), np.size(Z[i], 2)))
        zr = np.reshape(Z[i + 1], (np.size(Z[i + 1], 0), int(Z[i + 1].size / (np.size(Z[i + 1], 0)))))
        P = np.dot(zl, zr)
    Z_neq_out = np.reshape(P, (
    np.size(Z[0], 0), int(P.size / (np.size(Z[0], 0) * (np.size(Z[N - 1], 2)))), np.size(Z[N - 1], 2)))

    return Z_neq_out

