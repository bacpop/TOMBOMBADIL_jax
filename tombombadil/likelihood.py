
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .gtr import update_GTR


def gen_alpha(omega, A, pimat, pimult, pimatinv, scale):
    #print("A", A[7, ])
    mutmat = update_GTR(A, omega, pimult)
    #print("mutmat", mutmat)

    eps = 1e-4
    mutmat = mutmat + eps * jnp.eye(mutmat.shape[-1]) # add jitter to diagonal (avoids repeated eigenvalues --> eigenvectors are not uniquely defined --> gradient of eigenvectors is undefined / discontinuous --> nans in optimizer)
    # supposedly does not affect the model much (--> might need to confirm this later)

    #eigvals = jnp.linalg.eigvalsh(mutmat)
    #jax.debug.print(
    #    "eigvals[3] = {e}",
    #    e=eigvals[3],
    #)

    w, v = jnp.linalg.eigh(mutmat, UPLO='U') # computes eigen vectors (v) and values (w)
    #print(f"w.shape={w.shape}")
    #print(f"v.shape={v.shape}")
    E = 1 / (1 - 2 * scale * jnp.reshape(w, (61)))
    V_inv = jnp.matmul(jnp.reshape(v, (61, 61)), jnp.diag(E)) # TODO probably can be made more efficient

    # Create m_AB for each ancestral codon
    m_AB = jnp.zeros((61, 61))
    index = jnp.arange(0, 61, 1, jnp.uint16)
    #sqp = jnp.sqrt(pi_eq)
    for i in range(61):
        # Va = rep_matrix(row(V, i), 61)
        # Va should be matrix where rows are repeats of row i of V
        Va = jnp.repeat(v[i, :], 61).reshape(61, 61)
        # m_AB[i, ] = to_row_vector(rows_dot_product(Va, V_inv))
        #m_AB[i, :] = jnp.sum(Va * V_inv, axis=1) # previous version: possibly rows/columns swapped
        m_AB = m_AB.at[:, i].set(jnp.sum(Va * V_inv.T, axis=0))
    #print("m_AB", m_AB[7, ])
    #print("pimat", pimat)
    #print("pimatinv", pimatinv)
    # Add equilibrium frequencies 
    #m_AB = jnp.multiply(jnp.multiply(m_AB, pimatinv).T, pimat)
    m_AB = jnp.matmul(jnp.matmul(m_AB.T, pimatinv).T, pimat)
    #print("m_AB", m_AB[31, ]) 
    #print("m_AB2", m_AB[7, ])
    #### agrees with stan version up to here

    #print("m_AB", m_AB)
    #print((m_AB.max()))
    # Normalise by m_AA
    #m_AA = jnp.reshape(jnp.repeat(jnp.diagonal(m_AB), 61), (61, 61)) # Creates matrix with diagonals copied along each row
    #m_AA = m_AA.T
    #m_AB = jnp.maximum(jnp.divide(m_AB, m_AA) - jnp.eye(61, 61), 1.0e-06) # Makes min value 1e-6 (and sets diagonal, as -I makes this 0)
    #print("m_AB",m_AB.at[1,1])
    #print("m_AB",m_AB[:,1]/ m_AB[1,1])
    for i in range(61):
        m_AB = m_AB.at[:,i].set(jnp.true_divide(m_AB[:,i], m_AB[i,i]))
        m_AB = m_AB.at[i,i].set(1.0e-06)
    #    for j in range(61):
    #        if m_AB[i,j] < 0: 
    #            m_AB = m_AB.at[i,j].set(1.0e-06)
    
    m_AB = jnp.where(m_AB < 0, 1.0e-6, m_AB) # better with jax because if can lead to error "Attempted boolean conversion of traced array with shape bool[]"

    #plt.matshow(m_AB)
    #plt.show()
    m_AB = m_AB.T
    #print("m_AB3", m_AB[7, ])
    #print(jnp.diag(m_AB))
    #print("m_AA", m_AA[31, ])
    #print((m_AA.max())) 
    #print("m_AB", m_AB[31, ])
    #print((m_AB.max())) # appears to become zero here.
    muti = m_AB + jnp.eye(61, 61)
    #print(jnp.diag(muti))
    return muti


