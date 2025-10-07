
import numpy as np

from .gtr import update_GTR


def gen_alpha(omega, A, pimat, pimult, pimatinv, scale):
    mutmat = update_GTR(A, omega, pimult)

    w, v = np.linalg.eigh(mutmat, UPLO='U') # computes eigen vectors (v) and values (w)
    print(f"w.shape={w.shape}")
    print(f"v.shape={v.shape}")
    E = 1 / (1 - 2 * scale * np.reshape(w, (61)))
    V_inv = np.matmul(np.reshape(v, (61, 61)), np.diag(E)) # TODO probably can be made more efficient

    # Create m_AB for each ancestral codon
    m_AB = np.zeros((61, 61))
    index = np.arange(0, 61, 1, np.uint16)
    #sqp = jnp.sqrt(pi_eq)
    for i in range(61):
        # Va = rep_matrix(row(V, i), 61)
        # Va should be matrix where rows are repeats of row i of V
        Va = np.repeat(v[i, :], 61).reshape(61, 61)
        # m_AB[i, ] = to_row_vector(rows_dot_product(Va, V_inv))
        m_AB[i, :] = np.sum(Va * V_inv, axis=1)
    #print("m_AB", m_AB[31, ])
    # Add equilibrium frequencies 
    m_AB = np.multiply(np.multiply(m_AB, pimatinv).T, pimat)
    print("m_AB", m_AB[31, ]) 
    #print((m_AB.max()))
    # Normalise by m_AA
    m_AA = np.reshape(np.repeat(np.diagonal(m_AB), 61), (61, 61)) # Creates matrix with diagonals copied along each row
    m_AB = np.maximum(np.divide(m_AB, m_AA) - np.eye(61, 61), 1.0e-06) # Makes min value 1e-6 (and sets diagonal, as -I makes this 0)
    print("m_AA", m_AA[31, ])
    print((m_AA.max())) 
    print("m_AB", m_AB[31, ])
    print((m_AB.max())) # appears to become zero here.
    muti = m_AB + np.eye(61, 61)
    return muti


