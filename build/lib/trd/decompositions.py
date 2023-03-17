from math import prod
import numpy as np
from numpy.linalg import pinv
from trd.base import tensor_to_matrix, matrix_to_tensor, coreten2tr
from trd.functions import truncated_svd, tuple_ops, Msum_fun, Gfold, Gunfold, Pro2TraceNorm, mytenmat, tenmat_sb
from trd.ops import factor, cores_initialization, circshift, lrf_cores_initialization, Z_neq

def TRALS(tensor, tr_rank, maxiter=20):
    """returns all irreducible factors of x in vector
    T : ndarray
        d-dimensional tensor T
    r : 1D-array
        predefined TR-ranks r.
    Returns
    -------
    node : 1D-array
        Cores Zk
    """
    n = tensor.shape
    dims = len(n)
    tr_core = cores_initialization(n, tr_rank)
    for i in range(maxiter*dims):
        for j in range(1, dims + 1):
            rank_dim = len(tr_core)
            temp_tr_core = tr_core[j:] + tr_core[0:j]
            tr_total = np.copy(temp_tr_core[0])
            for k in range(rank_dim - 2):
                temp = np.copy(temp_tr_core[k + 1])
                shape_l = (int(tr_total.size / temp.shape[0]), temp.shape[0])
                tr_total_l = np.reshape(tr_total, shape_l, order='F').copy()
                shape_r = (temp.shape[0], temp.shape[1] * temp.shape[2])
                tr_total_r = np.reshape(temp, shape_r, order='F').copy()
                tr_total = np.dot(tr_total_l, tr_total_r)
            merge_tr_core = np.reshape(tr_total,(temp_tr_core[0].shape[0], int(tr_total.size / (temp_tr_core[0].shape[0] * temp_tr_core[rank_dim - 2].shape[2])), temp_tr_core[rank_dim - 2].shape[2]), order='F').copy()
            core_pinv = pinv(np.transpose(tensor_to_matrix(merge_tr_core, 2)))
            tr_core[j - 1] = matrix_to_tensor(np.dot(tensor_to_matrix(tensor, j), core_pinv), 2, tr_core[j - 1].shape)
    return tr_core


def TRSVD(tensor, ep=1e-5):

    n = tensor.shape
    dims = len(n)
    ep = ep / np.sqrt(dims)

    tr_rank = np.ones((dims), dtype='int')
    tr_core = cores_initialization(n, tr_rank.tolist())
    for i in range(dims - 1):
        T_ = tensor
        if i == 0:
            tensor = np.reshape(T_, (n[i], int(np.size(T_) / n[i])))
            u, s, v, rc = truncated_svd(tensor, ep)
            temp = np.cumprod(factor(rc))
            idx = int(np.abs(temp - np.sqrt(rc)) - 1)
            tr_rank[i + 1] = temp[idx]
            tr_rank[i] = rc / tr_rank[i + 1]
            u = u[:, 0:int(tr_rank[i] * tr_rank[i + 1])]
            u = np.reshape(u, (n[i], int(tr_rank[i + 1]), int(tr_rank[i])))
            tr_core[i] = np.transpose(u, (2, 0, 1))
            s = s[0:int(tr_rank[i] * tr_rank[i + 1])]
            v = v[:, 0:int(tr_rank[i] * tr_rank[i + 1])]
            v = np.dot(v, np.diag(s)).T
            v = np.reshape(v, (int(tr_rank[i + 1]), int(tr_rank[i]), np.prod(n[1:])))
            tensor = np.transpose(v, (0, 2, 1))
        else:
            m = int(tr_rank[i] * n[i])
            tensor = np.reshape(T_, (m, int(np.size(T_) / m)))
            u, s, v, r1 = truncated_svd(tensor, ep)
            tr_rank[i + 1] = np.maximum(r1, 1)
            u = u[:, 0:int(tr_rank[i + 1])]
            tr_core[i] = np.reshape(u, (int(tr_rank[i]), n[i], int(tr_rank[i + 1])))
            v = v[:, 0:int(tr_rank[i + 1])]
            s = s[0:int(tr_rank[i + 1])]
            tensor = np.dot(v, np.diag(s))
    tr_core[dims - 1] = np.reshape(tensor, (int(tr_rank[dims - 1]), n[dims - 1], int(tr_rank[0])))

    return tr_core, tr_rank


def b_computation(tr_core, tr_rank, od, n):
    b = tr_core[od[1]]
    dims = len(n)
    for k in range(2, dims):
        j = od[k]
        br = tr_core[j]
        br = br.reshape(tr_rank[j], int(br.size // tr_rank[j]))
        b = b.reshape(int(b.size // tr_rank[j]), tr_rank[j])
        b = b @ br
    b = b.reshape((tr_rank[od[1]], prod(tuple_ops(n, od[1:])), tr_rank[od[0]]), order='F')
    b = b.transpose((0, 2, 1))
    b = b.reshape((tr_rank[od[1]] * tr_rank[od[0]], prod(tuple_ops(n, od[1:]))), order='F')
    return b

def TRALSAR(tensor, maxiter=20, ep=1e-3):
    # n = tensor.shape
    # dims = len(n)
    c = tensor
    n = np.array(c.shape)
    dims = len(n)
    ratio = 0.1 / dims

    tr_rank = np.ones((dims), dtype='int')
    tr_core = cores_initialization(n, tr_rank.tolist())
    od = list(range(dims))

    for it in range(maxiter * dims):
        if it > 0:
            c = np.moveaxis(c, 0, -1)
            od = circshift(od)
        c = c.reshape(n[od[0]], c.size // n[od[0]], order='F')
        b = b_computation(tr_core, tr_rank, od, n)

        a0 = np.linalg.lstsq(b.T, c.T, rcond=None)[0].T
        err0 = np.linalg.norm(c - a0 @ b, ord='fro') / np.linalg.norm(c.flatten(), ord=2)

        a0 = a0.reshape(n[od[0]], tr_rank[od[1]], tr_rank[od[0]])
        tr_core[od[0]] = a0.transpose((2, 0, 1)).copy()
        tr_rank[od[1]] = tr_rank[od[1]] + 1
        node_od = np.mean(tr_core[od[1]].flatten()) + np.std(tr_core[od[1]].flatten()) * np.random.rand(n[od[1]],tr_rank[od[2]])
        # node_od = node_od.reshape(np.size(node_od, 1), -1, 1)
        # tr_core[od[1]] = np.concatenate((tr_core[od[1]], node_od), axis=0)
        tr_core[od[1]] = np.insert(tr_core[od[1]], tr_rank[od[1]]-1, values=node_od,axis=0)

        b = b_computation(tr_core, tr_rank, od, n)
        a1 = np.linalg.lstsq(b.T, c.T, rcond=None)[0].T
        err1 = np.linalg.norm(c - a1 @ b, ord='fro') / np.linalg.norm(c.flatten(), ord=2)

        if (err0 - err1) / (err0) > ratio * (err0 - ep) / (err0) and err0 > ep:
            a1 = a1.reshape((n[od[0]], tr_rank[od[1]], tr_rank[od[0]]), order='F')
            tr_core[od[0]] = a1.transpose((2, 0, 1))
            err0 = err1
            switch = 0
        else:
            tr_core[od[1]] = np.delete(tr_core[od[1]], tr_rank[od[1]] - 1, axis=0)
            tr_rank[od[1]] = tr_rank[od[1]] - 1
            switch = 1
        s = np.linalg.norm(tr_core[od[0]].flatten())
        tr_core[od[0]] = tr_core[od[0]] / s
        print('it: %d, err=%f' % (it, err0))
        if err0 < ep and it >= 2 * dims and switch == 1:
            break
        c = c.reshape(n[od])

    tr_core[od[0]] = tr_core[od[0]] * s

    return tr_core, tr_rank

def TRLRF(data, W, r, maxiter=300, K=1e0, ro=1.1, Lambda=5, tol=1e-6):
    T = data * W
    N = T.ndim
    S = T.shape
    S_1, S_2, S_3 = S
    r = np.array(r)
    X = np.random.rand(S_1, S_2, S_3)
    G = lrf_cores_initialization(S, r)
    M = np.zeros((N, 3), dtype=np.object)
    Y = np.zeros((N, 3), dtype=np.object)
    for i in range(N):
        G[i] = 1 * G[i]
        for j in range(3):
            M[i][j] = np.zeros(G[i].shape)
            Y[i][j] = np.sign(G[i])
    K_max = 10 ** 2
    Convergence_rec = np.zeros(maxiter)
    iter = 0
    while iter < maxiter:
        # update G
        for n in range(N):
            Msum = Msum_fun(M)
            Ysum = Msum_fun(Y)
            Q = tenmat_sb(Z_neq(G, n), 2)
            Q = Q.T
            G[n] = Gfold(np.dot((np.dot((Lambda * tenmat_sb(X, n + 1)), Q.T) + K * Gunfold(
                Msum[n], 1) + Gunfold(Ysum[n], 1)), np.linalg.pinv(
                (Lambda * (np.dot(Q, Q.T)) + 3 * K * np.eye(Q.shape[0], Q.shape[0])))), G[n].shape, 1)
            # update  M
            for j in range(3):
                Df = Gunfold(G[n] - Y[n][j] / K, j)
                M[n][j] = Gfold(Pro2TraceNorm(Df, 1 / K)[0], G[n].shape, j)
        # update X
        lastX = X
        X_hat = coreten2tr(G)
        X = X_hat
        X[W == 1] = T[W == 1]
        # update Y
        for n in range(N):
            for j in range(0, 3):
                Y[n, j] = Y[n, j] + K * (M[n, j] - G[n])
        K = min(K * ro, K_max)
        # evaluation
        G_out = G
        err_x = np.abs(
            (np.linalg.norm(lastX.T.flatten()) - np.linalg.norm(X.T.flatten())) / np.linalg.norm(X.T.flatten()))
        if err_x < tol:
            print('iteration stop at %f\n' % iter)
            break
        if iter % 100 == 0 or iter == 0:
            Ssum_G = 0  # singular value
            for j in range(N):
                _, vz1, __ = np.linalg.svd(mytenmat(G[0], 1, 1))
                _, vz2, __ = np.linalg.svd(mytenmat(G[0], 2, 1))
                _, vz3, __ = np.linalg.svd(mytenmat(G[0], 3, 1))

                Ssum_G = Ssum_G + np.sum(vz1.T.flatten()) + np.sum(vz2.T.flatten()) + np.sum(vz3.T.flatten())
            f_left = Ssum_G
            f_right = Lambda * (np.linalg.norm(X.T.flatten() - X_hat.T.flatten())) ** 2
            Convergence_rec[iter] = f_left + f_right
            print('TRLRF: Iter %f, Diff %d, Reg %d, Fitting %d' % (iter, err_x, f_left, f_right))
        iter = iter + 1

    return X