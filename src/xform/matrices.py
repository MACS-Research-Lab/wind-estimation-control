"""
Operations primarily on matrices. 
"""
import numpy as np
import scipy as sp



def ab_xform_from_pseudo_matrix(P_s: np.ndarray, P_t: np.ndarray=None, dt: float=1e-2):
    """
    Infer A,B matrices from pseudo matrices for source and target tasks.
    Also inver the least-squares transformation F_A, F_B between those
    matrices.

    Parameters
    ----------
    P_s : _type_
        Pseudo matrix for source task.
    P_t : np.ndarray
        Pseudo matrix for target task
    dt : _type_, optional
        Simulation time step, by default 1e-2

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A_s, B_s, A_t, B_t, F_A, F_B
    """
    nstates = P_s.shape[0]
    nactions = P_s.shape[1] - nstates
    # A -> P = [I + dt.A, dt.B] -> A = (P - I) / dt
    A_s = (P_s[:, :nstates] - np.eye(nstates)) / dt
    # B = P / dt
    B_s = P_s[:, nstates:] / dt
    if P_t is not None:
        A_t = (P_t[:, :nstates] - np.eye(nstates)) / dt
        # A_t = F_A A_s
        F_A = A_t @ np.linalg.pinv(A_s)
        # F_A = (A_t @ A_s.T) @ np.linalg.pinv(A_s @ A_s.T)
        B_t = P_t[:, nstates:] / dt
        # B_t = F_B B_s
        # F_B = (B_t @ B_s.T) @ np.linalg.pinv(B_s @ B_s.T)
        F_B = B_t @ np.linalg.pinv(B_s)
    else:
        A_t, B_t, F_A, F_B = None, None, None, None
    return A_s, B_s, A_t, B_t, F_A, F_B



def action_transform(x: np.ndarray, u: np.ndarray, A_s, B_s, F_A, F_B, A_t=None, B_t=None) -> np.ndarray:
    udims = u.ndim
    I = np.eye(len(A_s))
    x = np.atleast_2d(x)
    u = np.atleast_2d(u)
    if A_t is None and B_t is None:
        B_t = F_B @ B_s
        A_t = F_A @ A_s
        F_BB_ = B_t_ = np.linalg.pinv(B_t)
    # u_ = (F_BB_ @ ((I - F_A) @ A_s @ x.T + B_s @ u.T)).T
    u_ = (B_t_ @ ((A_s @ x.T - A_t @ x.T) + B_s @ u.T)).T

    if udims==1:
        return u_.squeeze()
    return u_



def pseudo_matrix_from_data(
    xu, x, mode='xu-x'
) -> np.ndarray:
    if mode=='xu-x':
        # P = (x @ xu.T) @ np.linalg.pinv(xu @ xu.T)
        P = (x) @ np.linalg.pinv(xu)
    return P



def nest_policy_xforms(state_xform0, action_xform0, state_xform1, action_xform1):
    #u0 = S0 x + A0 u
    #u1 = S1 x + A1 (S0 x + A0 u)
    #u1 = (S1 + A1 S0) x + A1 A0 u
    return (state_xform1 + action_xform1 @ state_xform0, action_xform1 @ action_xform0)



def dpolicy_dfa(
    A: np.ndarray, B: np.ndarray, F_B: np.ndarray,
    x: np.ndarray, u=None
) -> np.ndarray:
    x = np.atleast_2d(x).T # assume x argument was a row vector, converting to column
    F_BB_ = np.linalg.pinv(F_B @ B)
    d_dF_A = np.kron((A @ x).T, F_BB_)
    return -d_dF_A.T # last .T to convert to row vector format



def dpolicy_dfb(
    A: np.ndarray, B: np.ndarray, F_B: np.ndarray,
    x: np.ndarray, u: np.ndarray
) -> np.ndarray:
    x = np.atleast_2d(x).T
    u = np.atleast_2d(u).T
    F_BB_ = np.linalg.pinv(F_B @ B)
    return -np.kron((B @ u).T, F_BB_).T
    # return np.dot(-F_BB_ @ B, u.T).T # last .T to convert to row vector format



def dist_identity(mat: np.ndarray) -> float:
    """
    Distance of a matrix from identity.

    Parameters
    ----------
    mat : np.ndarray
        The matrix.

    Returns
    -------
    float
        norm of matrix - identity
    """
    mat = np.atleast_2d(mat)
    return np.linalg.norm(mat - np.eye(len(mat)))



def err_inv(mat: np.ndarray) -> float:
    """
    Inversion error of a matrix. Where

        err = ||M^{-1}M - I||
    
    Which for an invertible matrix is 0.

    Parameters
    ----------
    mat : np.ndarray
        The matrix.

    Returns
    -------
    float
        Inversion error.
    """
    mat = np.atleast_2d(mat)
    minv = np.linalg.pinv(mat)
    return np.linalg.norm(minv @ mat - np.eye(minv.shape[0]))



def basis_vectors(mat):
    # https://stackoverflow.com/a/42868363/4591810
    _, _, upper = sp.linalg.lu(mat)
    idx = [np.nonzero(upper[a])[0][0] for a in range(len(upper))]
    basis = mat[:, list(set(idx))]
    norms = np.linalg.norm(basis, axis=0)
    return basis / norms