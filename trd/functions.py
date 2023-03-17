import numpy as np
from trd.base import folding, unfolding, my_chop2

def n_mode_product(tensor, matrix, n):
    """n-mode product of a tensor with a matrix.
    Parameters
    ----------
    T : ndarray
    M : 2D-array
    n : int
        n mode
    Returns
    -------
    product : ndarray
        n-mode product
    """
    product_ = matrix @ unfolding(tensor, n)
    res_shape = list(tensor.shape)
    if matrix.ndim == 1:
        if len(res_shape) > 1:
            res_shape[n] = 1
        else:
            res_shape[1] = 1
        product = folding(product_, n, res_shape).squeeze()
    else:
        product = folding(product_, n, res_shape)
    return product

def inner_product(tensor1, tensor2):
    """inner product of two tensor with same shape
    Parameters
    ----------
    tensor1 : ndarray
    tensor2 : ndarray
    Returns
    -------
    product : ndarray
        n-mode product
    """
    product = np.sum(tensor1 * tensor2)
    return product

def n_mode_inner_product(tensor1, tensor2, n):
    """n mode inner product of two tensor
    Parameters
    ----------
    tensor1 : ndarray
    tensor2 : ndarray
    n : int
        n mode
    Returns
    -------
    product : ndarray
        n-mode product
    """
    common_size = int(np.prod(list(tensor1.shape)[-n:]))
    shape = list(tensor1.shape)[:-n] + list(tensor2.shape)[n:]
    product_ = np.reshape(tensor1, (-1, common_size)) @ np.reshape(tensor2, (common_size, -1))
    product = product_.reshape(shape)
    return product

def outer_products(tensors):
    """outer product of a list of tensors
    Parameters
    ----------
    Ts : list or tuple
        a list of tensors
    Returns
    -------
    product : ndarray
        outer_products
    """
    for i in range(len(tensors)):
        if i == 0:
            product = tensors[0]
            continue
        shape_before = product.shape + (1,) * tensors[i].ndim
        shape_after = (1,) * product.ndim + tensors[i].shape
        product = product.reshape(shape_before) * tensors[i].reshape(shape_after)
    return product

def truncated_svd(T, ep=1e-5):
    u, s, v = np.linalg.svd(T, full_matrices=False)
    v = v.T
    # s = np.diag(s_)
    rc = my_chop2(s, ep * np.linalg.norm(s, 2))
    return u,s,v,rc

def tuple_ops(tpl, idxs):
    new_list = []
    for idx in idxs:
        new_list.append(tpl[idx])
    return new_list

# generate tensor W by size and missing rate
def gen_W(S, mr):
    # np.prod(S)
    Omega_1 = np.random.permutation(np.prod(S))
    Omega = Omega_1[0:int(round((1 - mr) * np.prod(S)))].tolist()
    W_1 = np.zeros(np.prod(S))
    W_1[Omega] = 1
    W = W_1.reshape(S)
    return W

def RSE_fun(X, X_hat, W):
    # global RSE
    RSE_list = [0, 0, 0]
    err = X_hat.T.flatten() - X.T.flatten()
    RSE_list[0] = np.sqrt(np.sum(err ** 2) / np.sum((X.T.flatten()) ** 2))
    # RSE of observed entries
    X_hat_w = X_hat * W
    X_w = X * W
    err_w = X_hat_w.T.flatten() - X_w.T.flatten()
    RSE_list[1] = np.sqrt(np.sum(err_w ** 2) / np.sum((X_w.T.flatten()) ** 2))
    # RSE of missing entries
    Wr = np.logical_not(W)
    X_hat_wr = X_hat * Wr
    X_wr = X * Wr
    err_wr = X_hat_wr.T.flatten() - X_wr.T.flatten()
    RSE_list[2] = np.sqrt(np.sum(err_wr ** 2) / np.sum((X_wr.T.flatten()) ** 2))

    return RSE_list

def Msum_fun(M):
    N = np.size(M, 0)
    Msum_out = np.zeros((N,), dtype=np.object)
    for i in range(N):
        Msum_out[i] = M[i][0] + M[i][1] + M[i][2]
    return Msum_out

# fold core tensor to original size
def Gfold(Gm, SGt, n):
    if n == 0:
        Gt_out = np.reshape(Gm, (SGt[0], SGt[1], SGt[2]))
    elif n == 1:
        Gt_out = np.transpose(np.reshape(Gm, (SGt[1], SGt[0], SGt[2])), (1, 0, 2))
    elif n == 2:
        Gt_out = np.transpose(np.reshape(Gm, (SGt[2], SGt[0], SGt[1])), (1, 2, 0))
    else:
        print('wrong!')
    return Gt_out

# only for unfold core tensors to 3 different mode
def Gunfold(Gt, n):
    if n == 0:
        Gm = np.reshape(Gt, (Gt.shape[0], Gt.shape[1] * Gt.shape[2]))
    elif n == 1:
        Gm = np.reshape(np.transpose(Gt, (1, 0, 2)), (Gt.shape[1], Gt.shape[0] * Gt.shape[2]))
    elif n == 2:
        Gm = np.reshape(np.transpose(Gt, (2, 0, 1)), (Gt.shape[2], Gt.shape[0] * Gt.shape[1]))
    else:
        print('wrong!')
    return Gm

def Pro2TraceNorm(Z, tau):
    (m, n) = Z.shape
    if 2 * m < n:
        AAT = np.dot(Z, Z.T)
        S, Sigma2, D = np.linalg.svd(AAT)
        V = np.sqrt(Sigma2)
        tol = max(Z.shape) * np.finfo(max(V)).eps
        n = sum(V > max(tol, tau))
        mid = np.maximum(V[0:n - 1] - tau, 0) / V[0: n - 1]
        X = np.dot(np.dot(np.dot(S[:, 0:n - 1], np.diag(mid)), S[:, 0:n - 1].T), Z)
    elif m > 2 * n:
        X, n, Sigma2 = Pro2TraceNorm(Z.T, tau)
        X = X.T
    else:
        S, V, D = np.linalg.svd(Z)
        Sigma2 = V ** 2
        n = sum(V > tau)
        X = np.dot(np.dot(S[:, 0:n - 1], np.maximum(V[0:n - 1, 0:n - 1] - tau, 0)), D[:, 0:n - 1].T)

    return X, n, Sigma2

# from tensor to matrix
# 1 kolda mat, 2 tt mat, 3 tr mat
def mytenmat(X, n, type):
    S = X.shape
    N = len(S)
    if type == 1:  # the In x I1...In-1In+1...IN type
        if n == 1:
            Xn = np.reshape(X, (S[0], np.prod(S[1:])))
        elif n == N:
            Xn = np.transpose(np.reshape(X, (np.prod(S[0:-1]), S[N - 1])), (1, 0))
        else:
            arr_1 = np.array(n - 1)
            arr_2 = np.arange(0, n - 1)
            arr_3 = np.arange(n, N)
            arr = np.append(np.append(arr_1, arr_2), arr_3)
            Xn = np.reshape(np.transpose(X, arr), (S[n - 1], int(np.prod(S) / S[n - 1])))
    elif type == 2:  # the I1...In x In+1...IN type
        Xn = np.reshape(X, (np.prod(S[0:n]), int(X.size / np.prod(S[0:n]))))
    else:  # the In x In+1...INI1...In-1 type
        if n == 1:
            Xn = np.reshape(X, (S[0], int(X.size / S[0])))
        elif n == N:
            Xn = np.reshape(X, (int(X.size / S[N - 1]), S[N - 1]))
            Xn = np.transpose(Xn, (1, 0))
        else:
            Xn = np.reshape(X, (np.prod(S[0:n - 1]), int(X.size / np.prod(S[0:n - 1]))))
            Xn = np.transpose(Xn, (1, 0))
            Xn = np.reshape(Xn, (S[n - 1], int(X.size / S[n - 1])))
    return Xn

def tenmat_sb(X, k):
    S = X.shape
    N = len(S)
    if k == 1:
        X_sb_k = np.reshape(X, (S[0], int(X.size / S[0])))
    elif k == N:
        X_sb_k = np.reshape(X, (int(X.size / S[N - 1]), S[N - 1]))
        X_sb_k = np.transpose(X_sb_k, (1, 0))
    else:
        X = np.reshape(X, (np.prod(S[0:k - 1]), int(X.size / np.prod(S[0:k - 1]))))
        X = np.transpose(X, [1, 0])
        X_sb_k = np.reshape(X, (S[k - 1], int(X.size / S[k - 1])))
    return X_sb_k




