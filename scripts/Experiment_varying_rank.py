import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import numpy.linalg as la
import re
import scipy.sparse as sp
from scipy.stats import mode
from scipy.sparse.linalg import svds,norm
from scipy.sparse import csr_matrix,diags
from joblib import Parallel, delayed
from dataclasses import dataclass, asdict



# Load primes only once and cache them
file_path = Path(__file__).resolve().parents[1] / "data" / "list_of_primes.txt"


with open(file_path, 'r') as file:
    content = file.read()
    PRIMES_LIST = list(map(int, re.findall(r'\d+', content)))




# Define a dataclass to store experiment results
@dataclass
class ExperimentResult:
    expt_id: int
    rank: int   # this parameters specifies the rank 
    eigen_vector_index: int  # Track which singular vector's error is being computed
    true_eigenvalue: float # 
    approx_eigenvalue: float# 
    l2_true_meas: float # l2 norm of the true measurement of the eigenvector
    l2_approx_meas: float # l2 norm of the approximate measurement of the eigenvector
    rel_l2_error: float    # l2 error before inversion
    linf_error: float  #linf error before inversion
    l2_true_eig: float #l2 norm of the true eigenvector , this should be almost one for all of them 
    l2_approx_eig: float #l2 norm of the approximate (recovered) eigenvector
    rel_l2_inversion_error: float # l2 error after inversion
    linf_inversion_error: float #linf error after inversion
    index_difference: float # this keeps track of the number of entries in which the true eigenvector and the recovered eigevenctor differ in the first s positision
    beta: float # this compute the beta in the main theorem
    

########## Construct General matrix A, almost s-sparse ##########

####Gram_schmidt based on QR decomposition################
########## Construct General matrix A, almost s-sparse ##########

####Gram_schmidt based on QR decomposition################
def gram_schmidt(V):
    """
    Efficient Gram-Schmidt orthonormalization using QR decomposition.
    """
    Q, _ = np.linalg.qr(V)
    return Q


def generate_orthogonal_vectors_overlapping(N, r, s, times_of_nonzeros, ratio, decay_type):
    """
    Returns:
    - vectors: A list of r orthogonal sparse vectors (in CSR format).
    - random_numbers: A list of r positive random numbers.
    """
    
    # Calculate decay values
    if decay_type == 'expo':
        A_sval = 2.0 ** (-np.arange(r))  
    elif decay_type == 'lin':
        A_sval = np.linspace(1, 0, r, endpoint=False)  
    elif decay_type == 'gen':
        A_sval = 1.0 / (1 + 2 * np.arange(r))  
    else:
        raise ValueError("Invalid decay_type. Choose 'expo', 'lin', or 'gen'.")
    
    # Generate random orthogonal vectors
    eig_vecs = np.random.randn(s, r)
    orthogonal_vectors = gram_schmidt(eig_vecs)
    
    # Select common positions for pairwise orthogonality
    common_positions = np.random.choice(N, size=s, replace=False)
    
    # Calculate scaling factor
    c = ratio / (times_of_nonzeros ** 2) if times_of_nonzeros != 0 and ratio != 0 else 0
    
    # Initialize vectors
    vectors = []
    remaining_positions = np.setdiff1d(np.arange(N), common_positions)
    
    for i in range(r):
        # Assign random values to common positions
        common_values = orthogonal_vectors[:, i]
        common_row_indices = np.zeros(s, dtype=int)
        common_col_indices = common_positions
        
        # Assign non-zero entries to non-overlapping positions
        selected_positions = np.random.choice(remaining_positions, size=times_of_nonzeros * s, replace=False)
        selected_values = c * np.random.randn(times_of_nonzeros * s)
        selected_row_indices = np.zeros(times_of_nonzeros * s, dtype=int)
        selected_col_indices = selected_positions
        
        # Combine common and selected positions
        row_indices = np.concatenate([common_row_indices, selected_row_indices])
        col_indices = np.concatenate([common_col_indices, selected_col_indices])
        values = np.concatenate([common_values, selected_values])
        
        # Create sparse vector in COO format, convert to CSR, and normalize
        vec = sp.coo_matrix((values, (row_indices, col_indices)), shape=(1, N)).tocsr()
        norm_vec = vec / norm(vec)
        vectors.append(norm_vec)
        
        # Remove selected positions from remaining positions
        remaining_positions = np.setdiff1d(remaining_positions, selected_positions)
    
    return A_sval, vectors, common_positions

def generate_orthogonal_vectors_non_overlapping(N, r, s,times_of_nonzeros,ratio, decay_type):
    """
    Returns:
    - vectors: A list of r orthogonal sparse vectors (in CSR format).
    - random_numbers: A list of r positive random numbers.
    """
    if decay_type == 'expo':
        A_sval = 2.0 ** (-np.arange(r))  # Efficient vectorized exponential decay
    elif decay_type == 'lin':
        A_sval = np.linspace(1, 0, r, endpoint=False)  # Efficient linear decay
    elif decay_type == 'gen':
        A_sval = 1.0 / (1 + 2 * np.arange(r))  # Efficient generalized decay
    else:
        raise ValueError("Invalid decay_type. Choose 'expo', 'lin', or 'gen'.")
    
    # Generate an r by s array of random numbers
    eig_vecs = np.random.randn(s, r)
    #orthogonal_vectors = gram_schmidt(eig_vecs)
    
    #  Select the first r*s  positions for staircase style  orthogonality
    used_position = np.arange(r*s)
    

    if times_of_nonzeros == 0 or ratio ==0:
        c = 0
    else:
        c = ratio/(times_of_nonzeros**2)
    # Initialize the vectors as sparse matrices
    vectors = []
    remaining_positions = np.setdiff1d(np.arange(N), used_position)
    for i in range(r):
        largest_s_values = eig_vecs[:,i]
        common_row_indices = np.zeros(s, dtype=int)
        common_col_indices = used_position[i*s:(i+1)*s]
        
        # Assign 9s non-zero entries to non-overlapping positions
        selected_positions = np.random.choice(remaining_positions, size=times_of_nonzeros*s, replace=False)
        selected_values = c*np.random.randn(times_of_nonzeros*s)
        selected_row_indices = np.zeros(times_of_nonzeros*s, dtype=int)
        selected_col_indices = selected_positions
        
        # Combine common and selected positions
        row_indices = np.concatenate([common_row_indices, selected_row_indices])
        col_indices = np.concatenate([common_col_indices, selected_col_indices])
        values = np.concatenate([largest_s_values, selected_values])
        
        # Create a sparse vector in COO format and convert to CSR
        vec = sp.coo_matrix((values, (row_indices, col_indices)), shape=(1, N)).tocsr()
        norm_vec = vec/sp.linalg.norm(vec)
        vectors.append(norm_vec)
        
        # Remove the selected positions from the remaining positions
        remaining_positions = np.setdiff1d(remaining_positions, selected_positions)
    
    return A_sval , vectors

########## Find Primes (for construcitng P) ##########

# K = number of primes we want

def find_primes(K):
    out = PRIMES_LIST[:K]
    return out    


########## Optimized Prime Number Search ##########


def number_and_list_primes(N, s, smallest_prime):
    prime_0 = np.searchsorted(PRIMES_LIST, smallest_prime, side='right') # find the index of the smallest prime greater than smallest_prime
    primes = PRIMES_LIST[prime_0:]
    cumulative_product = 1
    alpha = 0
    # Calculate cumulative product until it exceeds N
    while cumulative_product < N and alpha < len(primes):
        cumulative_product *= primes[alpha]
        alpha += 1
    # Calculate the number of primes to be used
    number_of_primes = min(2 * alpha * s + 1, len(PRIMES_LIST))
    # create the list of primes used
    Primes_used = primes[:number_of_primes]
    return number_of_primes, Primes_used



########## Construct m matrix (based on prime numbers) ##########

# M is of size mP x N
# p_list is the list of primes we get from find_primes
# mP = sum of the primes


def M_matrix(primes, N):
    total_rows = sum(primes)
    val = 1/np.sqrt(len(primes))
    
    row_indices = []
    col_indices = []
    data = []
    
    row_offset = 0  # Track row index start
    
    for p in primes:
        # Assign 1s in a striped pattern for each prime
        for col in range(N):
            row = row_offset + (col % p)  # Cyclic row placement
            if row < row_offset + p:  # Ensure we stay within the block
                row_indices.append(row)
                col_indices.append(col)
                data.append(1)
        
        row_offset += p  # Move row start for next block
    
    # Create sparse matrix in COO format and convert to CSR for efficient operations
    M = sp.coo_matrix((data, (row_indices, col_indices)), shape=(total_rows, N))
    
    return val*M.tocsr()  # Convert to CSR format for efficient usage



'This code is used constructs the bit testing matrix B_N prime'
def build_BN_prime(N):
    d = int(np.ceil(np.log2(N)))  # d is the number of rows we expect in B_N'
    B = np.zeros((1 + d, N), dtype=int)
    B[0, :] = 1                     # we set the first rows to all 1s
    for j in range(N):
        bin_str = np.binary_repr(j, width=d)
        B[1:, j] = [int(b) for b in bin_str]
    B_comp = 1 - B
    B_prime = np.vstack([B, B_comp])
    return B_prime  # shape: (2 * (1 + log2(N)), N)

'This code compute the Khatri-Rao product of two matrices A and B'
def khatri_rao(A, B):
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B must have the same number of columns")
    I, K = A.shape
    J, _ = B.shape
    return np.vstack([np.kron(A[:, k], B[:, k]) for k in range(K)]).T




def compute_MAM_khatri(M, K, A_sval, A_svec):
    """
    This code computes  \tilde{M} A \tilde{M}^T where \tilde{M} = khatri_rao(M, B_N')
    and rescales by K * b, where b = 2(1 + ceil(log2 N))
    """
    r = len(A_sval)
    m, N = M.shape
    B_prime = build_BN_prime(N)
    b = B_prime.shape[0]  # b = 2(1 + log2 N)

    M_tilde = khatri_rao(M, B_prime)  # shape: (m * b, N)
    u_hat_dense = np.zeros((m * b, r))
    MAM = sp.csr_matrix((m * b, m * b))

    for j in range(r):
        vj = A_svec[j].T  # expected shape: (1, N) sparse row
        u_j = M_tilde @ vj.T  # (m*b x 1)
        u_hat_dense[:, j] = u_j.toarray().flatten()
        MAM += A_sval[j] * (u_j @ u_j.T)

    scale = 1 / (K * b)
    scale_u = 1 / np.sqrt(K * b)

    return MAM * scale, sp.csr_matrix(u_hat_dense) * scale_u

def safe_l2_norm(x, threshold=1e-17):
    "The function below compute the l2-norm of a vector but will set it to 0 if it's too small extermely tiny values"

    norm = np.linalg.norm(x)
    is_effectively_zero = norm < threshold
    return 0.0 if is_effectively_zero else norm

def safe_linf_norm(x, threshold=1e-15):
    norm = np.linalg.norm(x,np.inf)
    is_effectively_zero = norm < threshold
    return 0.0 if is_effectively_zero else norm

def MAM_inv_updated(N, S, M, K, y, s_guess):
    z = np.zeros(N)
    z_tmp = np.zeros(N)
    
    for n in S:
        col_n = M[:, n]
        nonzero_indices = col_n.nonzero()[0]
        nonzero_values = col_n.data
        
        # Sort nonzero values and get corresponding indices
        sorted_indices = nonzero_indices[np.argsort(nonzero_values)]
        top_k_indices = sorted_indices[-K:]
        top_k_values = y[top_k_indices]
        
        if np.all(top_k_values != 0):
            median_value = np.median(top_k_values)
            mode_value = mode(top_k_values, keepdims=True).mode[0]
            
            if median_value == mode_value:
                z_tmp[n] = median_value
    
    z_idx = np.argsort(np.abs(z_tmp))
    z[z_idx[-2*s_guess:]] = z_tmp[z_idx[-s_guess:]]    
    return z * np.sqrt(K)
# this function returns a list containing the position of the nonzeros in the kth column of M given the list of primes it was generated from
def nonzero_positions_in_column(primes,k):
    """Return the row indices of nonzero entries in the kth column of M_matrix."""
    row_indices = []
    row_offset = 0  # Starting row for each block
    
    for p in primes:
        row = row_offset + (k % p)  # Compute row index for this block
        if row < row_offset + p:  # Ensure within block
            row_indices.append(row)
        row_offset += p  # Move to next block
    
    return row_indices



def fastest_MAM_inv(N,S,primes,K,y,s_guess):
    'This iversion scheme does not need the matrix M'
    z = np.zeros(N)
    z_tmp = np.zeros(N)
    for n in S:
        idx_sort = nonzero_positions_in_column(primes,n)
        top_k_values = y[idx_sort[-K:]]  # Get the top-K values
        if np.all(top_k_values != 0):
            median_value = np.median(top_k_values)
            mode_value, _ = mode(top_k_values, keepdims=True)  # Compute the mode
            if median_value == mode_value[0]:
                z_tmp[n] = median_value
    z_idx = np.argsort(abs(z_tmp))
    for i in range(N-2*s_guess,N):
        z[z_idx[i]] = z_tmp[z_idx[i]]
    return z*np.sqrt(K)


def fast_create_sparse_matrix(prime_list, N):
    m = sum(prime_list)  # total number of rows
    K = len(prime_list)  # number of primes 
    total_entries = K * N  # number of nonzeros in M
    #nonzero_value = 1/np.sqrt(K)
    nonzero_value = 1
    
    # Preallocate row and column arrays
    row_indices = np.zeros(total_entries, dtype=np.int32)
    col_indices = np.zeros(total_entries, dtype=np.int32)
    
    nnz = 0  # Counter for non-zero elements
    row_offset = 0  # Start position for each prime block
    
    for p in prime_list:
        # Compute column indices directly using modulo operation
        col_vals = np.arange(N, dtype=np.int32)  # All columns
        row_vals = col_vals % p  # Compute row positions using modular arithmetic
        
        # Flatten and store indices
        num_entries = len(col_vals)
        row_indices[nnz:nnz + num_entries] = row_vals + row_offset
        col_indices[nnz:nnz + num_entries] = col_vals
        nnz += num_entries
        
        row_offset += p  # Move row start for next block
    
    # Trim excess space
    row_indices = row_indices[:nnz]
    col_indices = col_indices[:nnz]
    
    # Create sparse matrix in CSR format
    data = np.full(total_entries, nonzero_value, dtype=np.float32)
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(m, N))
    return A


# Optimized function to compute Mu_j without creating M

def efficient_M_sparse_vec_mult(primes, x):
    # Total number of rows in M (sum of the primes)
    total_rows = sum(primes)
    
    row_start = np.cumsum([0] + primes[:-1])

    # Initialize a result vector of the correct size
    result_vector = np.zeros(total_rows)
    
    # For each nonzero element in x
    for idx in x.indices:
        # Value of the nonzero entry in x
        value = x.data[np.where(x.indices == idx)[0][0]]

        # For each prime block, determine affected rows in Mx^T
        for block_id, p in enumerate(primes):
            row_within_block = idx % p  # Row index inside the block
            global_row = row_start[block_id] + row_within_block
            
            # Add the value to the correct entry in the result vector y
            result_vector[global_row] += value
    
    # Convert the result to a CSR matrix (sparse format)
    MxT = csr_matrix(result_vector)
    
    return MxT


'''
Here define various way to compute the MAM matrix
''' 
def compute_MAM_fast(M,K, A_sval, A_svec):
    r = len(A_sval)
    m, _ = M.shape
    u_hat_dense = np.zeros((m, r)) 
    MAM = sp.csr_matrix((m, m))  
    for j in range(r):
        #u_j = M @ (D.multiply(A_svec[j].T))
        u_j = M @ A_svec[j].T
        u_hat_dense[:, j] = u_j.toarray().flatten()  
        MAM += A_sval[j] * (u_j @ u_j.T)
    return MAM*(1/K), sp.csr_matrix(u_hat_dense)*(1/np.sqrt(K))  


def compute_MAM_fast_stair(M,K,s, D, A_sval, A_svec):
    r = len(A_sval)
    m, _ = M.shape
    u_hat_dense = np.zeros((m, r))  
    MAM = sp.csr_matrix((m, m))  
    for j in range(r):
        u_j = M[:, j * s:(j + 1) * s] @ (D[j * s:(j + 1) * s].multiply(A_svec[:,j]))
        norm_j = norm(u_j)
        u_hat_dense[:, j] = (u_j / norm_j).toarray().flatten() 
        MAM += A_sval[j] * (u_j @ u_j.T)
    return MAM*(1/K), sp.csr_matrix(u_hat_dense)*(1/np.sqrt(K))


# This is the most efficient way of computing MAM because it avoids using M

def MAM_fastest(primes_used, A_sval, A_svec):
    K = len(primes_used)
    r = len(A_sval)
    m = int(np.sum(primes_used))
    u_hat_dense = np.zeros((m, r))  
    MAM = sp.csr_matrix((m, m))
    for j in range(r):
        u_j = efficient_M_sparse_vec_mult(primes_used, A_svec[j])
        norm_j = norm(u_j)
        u_hat_dense[:, j] = (u_j / norm_j).toarray().flatten() 
        MAM += A_sval[j] * (u_j.T @ u_j)
    return MAM*(1/K), sp.csr_matrix(u_hat_dense)*(1/np.sqrt(K))


def khatri_rao(A, B):
    # Thus function computes the Khatri-Rao product of two matrices A and B.
    #A: shape (m, n)
    #B: shape (p, n)
    #Returns C : shape (m*p,n)
    assert A.shape[1] == B.shape[1], "A and B must have the same number of columns"
    
    m, n = A.shape
    p = B.shape[0]
    C = np.zeros((m * p, n))

    for j in range(n):
        C[:, j] = np.kron(A[:, j], B[:, j])
    
    return C


def bit_testing_matrix(N):
    # Construct the bit testing matrix B_N of shape (1 + ceil(log2(N)), N).
    # Each column j (1-based) is [1; binary(j-1)].
    
    num_bits = int(np.ceil(np.log2(N)))
    B = np.zeros((1 + num_bits, N), dtype=int)

    for j in range(N):
        binary_repr = np.array(list(np.binary_repr(j, width=num_bits)), dtype=int)
        rev_binary_rep = binary_repr[::-1]
        B[:, j] = np.concatenate(([1], rev_binary_rep))
    
    return B

def extended_bit_testing_matrix(N):
    # This function constructs the extended bit testing matrix B_N' of shape (2 * (1 + ceil(log2(N))), N).
    #Each column is the concatenation of B_N[:, j] and its complement.
    B = bit_testing_matrix(N)
    B_complement = 1 - B  # binary complement
    B_extended = np.vstack([B, B_complement])
    return B_extended


""""
This section of the code is mainly focused on computing the errors

"""


def top_k_singular_values_and_vectors(A, k):
    "This function computes the first k singular values and vectors of any matrix A in sparse format"
    u, s, _ = svds(A, k=k, which='LM', return_singular_vectors="u")
    return np.flip(s), np.flip(u, axis=1)


def rel_error_computation_top_6(p_list, K, N, u_hat, A_svec, MAM_left, s_guess, S):
    " Pre-allocate arrays for efficiency"
    l2_errors = np.zeros(6)
    l2_norms_true_eig = np.zeros(6)
    l2_norms_recov_eig = np.zeros(6)
    linf_errors = np.zeros(6)
    l2_norms_meas_eig = np.zeros(6)
    l2_norms_approx_meas_eig = np.zeros(6)
    l2_inversion_errors = np.zeros(6)
    linf_inversion_errors = np.zeros(6)
    diff_in_pos = np.zeros(6)

    for k in range(6):
        u_k = u_hat[:, k].toarray().ravel()  # Convert only once per iteration

        # Compute errors before inversion
        MAM_k = MAM_left[:, k]
        diff = MAM_k - u_k
        diff_plus = MAM_k + u_k
        true_meas = la.norm(MAM_k, 2) 
        approx_meas = la.norm(u_k,2)
        err_2 = min(safe_l2_norm(diff), safe_l2_norm(diff_plus))
        err_inf = min(safe_linf_norm(diff), safe_linf_norm(diff_plus))

        l2_errors[k] = err_2 / true_meas
        linf_errors[k] = err_inf
        l2_norms_meas_eig[k]  = true_meas
        l2_norms_approx_meas_eig[k] = approx_meas
        # Compute errors after inversion
        svec_inverted = fastest_MAM_inv(N, S, p_list, K, u_k, s_guess)  # Efficient inversion
        A_svec_k = A_svec[k].toarray().ravel()  # Convert once
        number_of_overlap = len(np.intersect1d(S,np.argsort(abs(svec_inverted))[-s_guess:]))

        #diff_in_position_k = len(np.intersect1d(S,largest_entries_recovered))
        inv_diff = svec_inverted - A_svec_k
        inv_diff_plus = svec_inverted + A_svec_k
        norm_true_eig = la.norm(A_svec_k)
        norm_recov_eig = la.norm(svec_inverted,2)
        inv_err_2 = min(safe_l2_norm(inv_diff), safe_l2_norm(inv_diff_plus))
        inv_err_inf = min(safe_linf_norm(inv_diff), safe_linf_norm(inv_diff_plus))

        l2_inversion_errors[k] = inv_err_2 / norm_true_eig
        linf_inversion_errors[k] = inv_err_inf
        l2_norms_true_eig[k] = norm_true_eig
        l2_norms_recov_eig[k] = norm_recov_eig
        diff_in_pos[k] = number_of_overlap

    return l2_norms_meas_eig.tolist(),l2_norms_approx_meas_eig.tolist() ,l2_errors.tolist(), linf_errors.tolist(), l2_norms_true_eig.tolist(), l2_norms_recov_eig.tolist(),l2_inversion_errors.tolist(), linf_inversion_errors.tolist(),diff_in_pos


def fast_error_computation(N,r,primes_used,u_hat, A_svec, MAM_left,s_guess,S):
    K = len(primes_used)
    l2_errors = []
    linf_errors = []
    l2_inversion_errors = []
    linf_inversion_errors = []
  
    for k in range(4):
        #u_j = A_svec[k].toarray().flatten()
        err_2 = min(la.norm(u_hat[:, k].toarray().flatten() - MAM_left[:, k], 2), la.norm(u_hat[:, k].toarray().flatten() + MAM_left[:, k], 2))
        err_inf = min(la.norm(u_hat[:, k].toarray().flatten() - MAM_left[:, k], np.inf), la.norm(u_hat[:, k].toarray().flatten() + MAM_left[:, k], np.inf))
        l2_errors.append(err_2)
        linf_errors.append(err_inf)
        # Compute errors after inversion
        #svec_inverted = MAM_inv_updated(N, S, M, K, u_hat[:, k].toarray().flatten(), s_guess)
        svec_inverted = fastest_MAM_inv(N,S,primes_used,K,u_hat[:, k].toarray().flatten(),s_guess)
        #index_j = np.nonzero(svec_inverted)
    
        inv_err_2 = min(la.norm(svec_inverted - A_svec[k].toarray().flatten(), 2),la.norm(svec_inverted + A_svec[k].toarray().flatten(), 2))
    
        inv_err_inf = min(la.norm(svec_inverted - A_svec[k].toarray().flatten(), np.inf),la.norm(svec_inverted + A_svec[k].toarray().flatten(), np.inf))
        l2_inversion_errors.append(inv_err_2)
        linf_inversion_errors.append(inv_err_inf)
    return l2_errors, linf_errors, l2_inversion_errors, linf_inversion_errors


def compute_f(x, B, s):
    x_s = np.zeros_like(x)
    top_s_indices = np.argsort(np.abs(x))[-s:]
    x_s[top_s_indices] = x[top_s_indices]
    
    norm1_term = (1/s) * np.linalg.norm(x - x_s, 1)
    norm_inf_term = np.linalg.norm(x - B, np.inf)
    denominator = norm1_term + norm_inf_term
    
    f_values = np.abs(x) / denominator if denominator != 0 else np.zeros_like(x)
    return f_values


def compute_top_only_f(x, B, s):
    x_s = np.zeros_like(x)
    top_s_indices = np.argsort(np.abs(x))[-s:]
    x_s[top_s_indices] = x[top_s_indices]
    
    norm1_term = (1/s) * np.linalg.norm(x - x_s, 1)
    norm_inf_term = np.linalg.norm(x - B, np.inf)
    denominator = norm1_term + norm_inf_term

    f_values = np.abs(x) / denominator if denominator != 0 else np.zeros_like(x)
    top_s_values = f_values[np.argsort(f_values)[-s:]]
    return top_s_indices,top_s_values

def compute_ratio(x,s):
    x_s = np.zeros_like(x)
    top_s_indices = np.argsort(np.abs(x))[-s:]
    x_s[top_s_indices] = x[top_s_indices]

    numerator = np.linalg.norm(x_s,1)
    denominator = np.linalg.norm(x, 1)
    
    ratio = numerator / denominator if denominator != 0 else 0
    return ratio

"""The following function will be used to increase the size of the list of primes that we start the experiment with"""
def add_primes_tolist(Intial_list_of_primes, end_position_initial_list,number_to_add):
    if number_to_add == 0:
        new_list = Intial_list_of_primes
    else:
        to_be_added = PRIMES_LIST[end_position_initial_list+1:end_position_initial_list+number_to_add+1]
        new_list = Intial_list_of_primes + to_be_added
    return new_list

###################### Section for Experiment##################################


def append_results_to_csv(results, csv_path):
    """Append multiple experiment results to CSV at once."""
    df = pd.DataFrame(results)
    df.columns = [col.capitalize() for col in df.columns]
    file_exists = csv_path.exists()
    df.to_csv(csv_path, mode='a', header=not file_exists, index=False)

def run_single_experiment(N, r,s,times_of_nnz,ratio, decay_type,support_type,p_list,M,run):
    """Run a single experiment iteration for a given sparsity level s and run index."""
    
    if support_type == 'disjoint':
        A_sval, A_svec = generate_orthogonal_vectors_non_overlapping(N, r, s, times_of_nnz, ratio, decay_type)
        S = range(int(4 * r * s))
    else:
        A_sval, A_svec, S = generate_orthogonal_vectors_overlapping(N, r, s, times_of_nnz, ratio, decay_type)

    K =len(p_list)
    #m = np.sum(p_list)
    #M = fast_create_sparse_matrix(p_list,N)    
    MAM, u_hat = compute_MAM_fast(M, K, A_sval, A_svec)
    MAM_vals, MAM_left = top_k_singular_values_and_vectors(MAM, r)
    l2_norms_meas_eig,l2_norms_approx_meas_eig ,l2_errors, linf_errors, l2_norms_true_eig, l2_norms_recov_eig,l2_inversion_errors, linf_inversion_errors, diff_position = rel_error_computation_top_6(p_list, K, N, u_hat, A_svec, MAM_left, s, S)
    
    results = []
    for k in range(6):
        results.append({
            'rank': r,    
            'eigen_vector_index': k,
            'true_eigenvalue': A_sval[k], 
            'approx_eigenvalue':  MAM_vals[k],
            'l2_true_meas': l2_norms_meas_eig[k],
            'l2_approx_meas': l2_norms_approx_meas_eig[k], 
            'rel_l2_error': l2_errors[k],   
            'linf_error': linf_errors[k],  
            'l2_true_eig': l2_norms_true_eig[k],  
            'l2_approx_eig': l2_norms_recov_eig[k],
            'rel_l2_inversion_error': l2_inversion_errors[k], 
            'linf_inversion_error': linf_inversion_errors[k], #linf error after inversion
            'index_difference': diff_position[k] # this keeps track of the number of entries in which the true eigenvector and the recovered eigevenctor differ in the first s positision
        })
    return results

def run_experiment_varying_rank(N, s, decay_type, times_of_nnz, ratio, support_type, r_list, p_list, runs, csv_path):
    """Run the experiment with parallel execution."""
    M = fast_create_sparse_matrix(p_list, N)
    results = Parallel(n_jobs=4, verbose=4, prefer="processes")(  
        delayed(run_single_experiment)(N, r,s,times_of_nnz,ratio, decay_type,support_type,p_list,M,run)
        for r in r_list for run in range(runs)
    )
    df = pd.DataFrame([item for sublist in results for item in sublist])
    df.to_csv(csv_path, mode='a', index=False, header=not Path(csv_path).exists())
    print('Experiment data generated in parallel!')


def plot_experiment(csv_path: Path, fig_dir: Path, param_str: str, N, s, m, K, r_list):
    """Plot errors vs rank for the first 4 singular vectors."""
    df = pd.read_csv(csv_path)
    df.columns = [col.lower() for col in df.columns]
    df_filtered = df[df['eigen_vector_index'] < 4]
    mean_errors = df_filtered.groupby(['rank', 'eigen_vector_index']).mean().reset_index()
    #title = f'N={N}, s={s}, m={m}, K={K}'
    file_prefix = f"{param_str}_m={m}"
    markers = ['o', 's', 'D', '^']

    def plot_metric(metric, ylabel, filename):
        plt.figure(figsize=(15, 12))
        ranked = ['st eigenvector', 'nd eigenvector', 'rd eigenvector', 'th eigenvector']
        for k in range(4):
            subset = mean_errors[mean_errors['eigen_vector_index'] == k]
            marker = markers[k]
            ranks = ranked[k]
            plt.plot(subset['rank'], subset[metric], label=f'{k+1} {ranks}', marker = marker,linewidth=4)
        plt.xlabel('Rank (r)',fontsize = 20)
        plt.ylabel(ylabel,fontsize= 20)
        plt.title(f'{ylabel} vs Rank')
        plt.xticks(np.arange(min(r_list), max(r_list), 2),size = 20)
        plt.yticks(size = 20)
        plt.legend(fontsize = 25)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fig_dir / f'{file_prefix}{filename}_mean.png')
        plt.close()
    
    plot_metric('rel_l2_error', 'Relative L2 Error Before Inversion', 'rel_l2_error_before_inversion')
    plot_metric('linf_error', 'L-infinity Error Before Inversion', 'linf_error_before_inversion')
    plot_metric('rel_l2_inversion_error', 'Relative L2 Error After Inversion', 'rel_l2_error_after_inversion')
    plot_metric('linf_inversion_error', 'L-infinity Error After Inversion', 'linf_error_after_inversion')
    print('Plots saved')


def main():
    N = 100 # Matrix size
    s = 9  # Sparsity
    decay_type = 'expo'  # Decay type
    times_of_nnz = 0  # Times of nonzeros
    ratio = 0  # Ratio
    support_type = 'disjoint'  # Support type

    #K = 70  # Number of primes
    r_list = [6,7]  # List of rank values
    K,p_list  = number_and_list_primes(100,2, 20)
    smallest_prime = p_list[0]
    runs = 1  # Number of runs per rank value

    # Create output directories
    fig_dir = Path(__file__).resolve().parents[1] / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
 
    param_str = f'N={N}_s={s}_decay={decay_type}_nnz={times_of_nnz}_ratio={ratio}_support={support_type}_K={K}_smallest_primes={smallest_prime}_r={"_".join(map(str, r_list))}_runs={runs}'
    csv_path = fig_dir / f'results_rank_{param_str}.csv'

    # Run the experiment
    run_experiment_varying_rank(N, s, decay_type, times_of_nnz, ratio, support_type, r_list, p_list, runs, csv_path)
    # Compute m (number of rows in M)
    #p_list = find_primes(K)
    m = np.sum(p_list)
    # Plot the results
    plot_experiment(csv_path, fig_dir,param_str,N, s, m, K, r_list)
if __name__ == '__main__':
    main()
