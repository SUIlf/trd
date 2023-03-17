import numpy as np

def unfolding(tensor, n=0):
    """Mode-n unfolding of a tensor
    Parameters
    ----------
    tensor : ndarray
        Tensor to be unfolded
    n : int
        mode n to split
    Returns
    -------
    unfolded_tensor : ndarray
        unfolded tensor
    """
    T_= np.moveaxis(tensor, n, 0)
    unfolded_tensor = np.reshape(T_, (tensor.shape[n], -1))
    return unfolded_tensor

def folding(unfolded_tensor, n, shape):
    """Refolds the Mode-n unfolded tensor
    Parameters
    ----------
    unfolded_tensor : ndarray
    n : int
    shape : tuple or list
    Returns
    -------
    folded_tensor : ndarray
        Folded tensor
    """
    if type(shape) == tuple:
        T_shape = list(shape)
    else:
        T_shape = shape
    mode_dim = T_shape.pop(n)
    T_shape.insert(0, mode_dim)
    T_ = np.reshape(unfolded_tensor, T_shape)
    folded_tensor = np.moveaxis(T_, 0, n)
    return folded_tensor

def tensor_to_matrix(tensor, n):
    """Matricize a tensor
    Parameters
    ----------
    tensor : ndarray
    n : int
    Returns
    -------
    matrix : 2D-array
        Matricization of a tensor
    """
    _matrix = tensor.transpose(np.append(np.arange(n - 1, tensor.ndim), np.arange(0, n - 1)))
    matrix = _matrix.reshape(tensor.shape[n-1], int(tensor.size/tensor.shape[n-1]), order='F').copy()
    # print("Type: %d" %(mat_type), ", Reshape at mode-%d" %(n), ", Transpose index:", arr, ", Matrix size: %u x %u" %(mat.shape[0], mat.shape[1]))
    return matrix

# reshape the "matricized tensor" to tensor
def matrix_to_tensor(matrix, n, shape):
    """Reshape the "matricized tensor" to tensor
    Parameters
    ----------
    matrix : 2D-array
    n : int
    shape : tuple
    Returns
    -------
    tensor : ndarray
        tesnorized tensor
    """
    array_product_b = np.prod(shape[n-1:])
    array_product_f = np.prod(shape[0:n-1])
    arr = np.append(array_product_b, array_product_f).astype('int')
    __tensor = np.reshape(matrix, arr, order='F').copy()
    _tensor = __tensor.transpose(1, 0).copy()
    tensor = np.reshape(_tensor,shape,order='F').copy()

    return tensor

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

def coreten2tr(Z):
    N = Z.size
    S = np.zeros(3, dtype=object)
    for i in range(N):
        S[i] = Z[i].shape[1]
    P = Z[0]
    for i in range(1, N):
        L = np.reshape(P, (int(P.size / Z[i - 1].shape[2]), Z[i - 1].shape[2]))
        R = np.reshape(Z[i], (Z[i].shape[0], int(S[i] * Z[i].shape[2])))
        P = np.dot(L, R)
    P = np.reshape(P, (Z[0].shape[0], np.prod(S), Z[N - 1].shape[2]))
    P = np.transpose(P, [1, 2, 0])
    P = np.reshape(P, (np.prod(S), int(Z[0].shape[0] * Z[0].shape[0])))
    temp = np.eye(Z[0].shape[0], Z[0].shape[0])
    P = np.dot(P, (temp.T.flatten()))
    X = np.reshape(P, S)
    return X

