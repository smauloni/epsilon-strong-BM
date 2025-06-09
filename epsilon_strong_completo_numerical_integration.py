import math
import random
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt

# Constants for alternating series (can be adjusted for precision/performance)
MAX_ITER_ZETA = 100  # Max terms for zeta series (for direct zeta calculation if used)
TOLERANCE_ZETA = 1e-12 # Tolerance for zeta series convergence (for direct zeta calculation)
MAX_ITER_COMPARISON = 20 # Max iterations for the alternating series comparison loop
QUAD_EPS_ABS = 1e-9 # Absolute tolerance for scipy.integrate.quad
QUAD_EPS_REL = 1e-9 # Relative tolerance for scipy.integrate.quad
ROOT_FINDER_XTOL = 1e-7 # Tolerance for root finding (brentq)
MAX_REFINE_ITER = 10 # Max refinement attempts for a single layer extremum

# --- Helper Functions for Zeta Calculation (Section 4) ---

def sigma_bar_j(j, x, y, delta, xi, l_duration):
    """
    Calculates sigma_bar_j as per eq (4.2) (first part).
    l_duration is 'l' in the paper's notation.
    j: term index
    x, y: start and end points of the bridge segment
    delta: U - L
    xi: L (for the first sigma_bar_j term) or -U (for the second)
    l_duration: duration of the bridge segment
    """
    if l_duration <= 1e-12: 
        return 0.0 
    exponent = - (2.0 / l_duration) * (delta * j + xi - x) * (delta * j + xi - y)
    try:
        res = math.exp(exponent)
        return res if math.isfinite(res) else (float('inf') if exponent > 0 else 0.0)
    except OverflowError:
        return float('inf') if exponent > 0 else 0.0


def tau_bar_j(j, x, y, delta, l_duration):
    """
    Calculates tau_bar_j as per eq (4.2) (second part).
    l_duration is 'l' in the paper's notation.
    j: term index
    x, y: start and end points of the bridge segment
    delta: U - L
    l_duration: duration of the bridge segment
    """
    if l_duration <= 1e-12: 
        return 0.0 
    exponent = - (2.0 * j / l_duration) * (delta**2 * j + delta * (x - y))
    try:
        res = math.exp(exponent)
        return res if math.isfinite(res) else (float('inf') if exponent > 0 else 0.0)
    except OverflowError:
        return float('inf') if exponent > 0 else 0.0

def get_alternating_zeta_sum_term(zeta_params, term_index_k):
    """
    Calculates the k-th pair of alternating series sums for zeta:
    S_{2k-1} (upper bound for zeta) and S_{2k} (lower bound for zeta).

    Args:
        zeta_params (tuple): (L, U, l_duration, x, y) for the specific zeta_i.
        term_index_k (int): Corresponds to 'k' in the paper's notation S_{2k-1} and S_{2k}.
                             Must be >= 1.

    Returns:
        tuple: (s_2k_minus_1, s_2k)
    """
    L, U, l_duration, x, y = zeta_params

    # Check if path is already outside L,U or if L,U are valid
    if U <= L + 1e-9: # Invalid U,L or U very close to L
        if abs(U-L) < 1e-9 and abs(x-L) < 1e-9 and abs(y-L) < 1e-9: # Path is on the line L=U
            return 0.0, 0.0 # Zeta = 0
        else: # Path must exit if L=U and x or y is not on L
            return 1.0, 1.0 # Zeta = 1

    if not (L < x < U and L < y < U): # Path starts or ends outside, or on boundary
        # If x or y is exactly L or U, it's tricky. The paper's formula is for L < x,y < U.
        # If x,y are on or outside bounds, zeta is 1.
        return 1.0, 1.0 

    delta = U - L # Should be > 0 here

    if term_index_k < 1:
        term_index_k = 1

    current_sum_sigma_minus_tau = 0.0
    s_2k_minus_1_for_current_j = 0.0 
    s_2k_for_current_j = 0.0         

    for j_loop_idx in range(1, term_index_k + 1):
        sigma_j_term1 = sigma_bar_j(j_loop_idx, x, y, delta, L, l_duration)
        sigma_j_term2 = sigma_bar_j(j_loop_idx, -x, -y, delta, -U, l_duration)
        sigma_j = sigma_j_term1 + sigma_j_term2

        tau_j_term1 = tau_bar_j(j_loop_idx, x, y, delta, l_duration)
        tau_j_term2 = tau_bar_j(j_loop_idx, -x, -y, delta, l_duration)
        tau_j = tau_j_term1 + tau_j_term2
        
        s_2k_minus_1_for_current_j = current_sum_sigma_minus_tau + sigma_j
        s_2k_for_current_j = s_2k_minus_1_for_current_j - tau_j
        
        current_sum_sigma_minus_tau += (sigma_j - tau_j)

    final_s_upper = s_2k_minus_1_for_current_j 
    final_s_lower = s_2k_for_current_j        

    final_s_upper = max(0.0, min(1.0, final_s_upper if math.isfinite(final_s_upper) else 1.0))
    final_s_lower = max(0.0, min(1.0, final_s_lower if math.isfinite(final_s_lower) else 0.0))
    
    if final_s_lower > final_s_upper + 1e-9: 
        final_s_lower = final_s_upper 
    return final_s_upper, final_s_lower

def zeta(L, U, l_duration, x, y):
    """
    Calculates zeta(L, U; l, x, y) by summing the series.
    Used by gamma_func and subsequently by the simplified rho and beta.
    """
    if U <= L + 1e-9:
        return 0.0 if abs(U-L) < 1e-9 and abs(x-L) < 1e-9 and abs(y-L) < 1e-9 else 1.0
    if not (L < x < U and L < y < U):
        return 1.0

    delta = U - L
    current_zeta_sum = 0.0
    
    for j_term in range(1, MAX_ITER_ZETA + 1):
        sigma_j_term1 = sigma_bar_j(j_term, x, y, delta, L, l_duration)
        sigma_j_term2 = sigma_bar_j(j_term, -x, -y, delta, -U, l_duration)
        sigma_j = sigma_j_term1 + sigma_j_term2

        tau_j_term1 = tau_bar_j(j_term, x, y, delta, l_duration)
        tau_j_term2 = tau_bar_j(j_term, -x, -y, delta, l_duration) 
        tau_j = tau_j_term1 + tau_j_term2
        
        term_val = sigma_j - tau_j
        if not math.isfinite(term_val): 
            break

        old_sum = current_zeta_sum
        current_zeta_sum += term_val
        
        if j_term > 1 :
            if old_sum != 0 and abs(term_val / old_sum) < TOLERANCE_ZETA: break
            elif abs(term_val) < TOLERANCE_ZETA * 1e-2: break
            elif term_val == 0 and old_sum == 0: break
            
    return max(0.0, min(1.0, current_zeta_sum if math.isfinite(current_zeta_sum) else 1.0))

def gamma_func(L, U, l_duration, x, y):
    # Ensure L < U for gamma/zeta calculation; if L >= U, gamma is 0 unless path is impossible
    if L >= U - 1e-9 : # If L is very close to U or greater
        # If path is x=y=L=U, it stays in, gamma = 1. Otherwise, gamma = 0.
        is_on_line = abs(L-U) < 1e-9 and abs(x-L) < 1e-9 and abs(y-L) < 1e-9
        return 1.0 if is_on_line else 0.0
    return 1.0 - zeta(L, U, l_duration, x, y)

def beta_func(L_low, L_high, U_low, U_high, l_dur, x_start, y_end):
    """
    Calculates beta(L_low, L_high, U_low, U_high; l, x, y) as per eq (4.8).
    P(L_low < min < L_high, U_low < max < U_high).
    Requires L_low < L_high, U_low < U_high, and L_high <= U_low.
    """
    if not (L_low < L_high - 1e-9 and \
            U_low < U_high - 1e-9 and \
            L_high <= U_low + 1e-9):
        return 0.0

    term1 = gamma_func(L_low,  U_high, l_dur, x_start, y_end)
    term2 = gamma_func(L_high, U_high, l_dur, x_start, y_end)
    term3 = gamma_func(L_low,  U_low,  l_dur, x_start, y_end)
    term4 = gamma_func(L_high, U_low,  l_dur, x_start, y_end)
    
    beta_val = term1 - term2 - term3 + term4
    return max(0.0, min(1.0, beta_val if math.isfinite(beta_val) else 0.0))


def rho(L_downarrow, L_uparrow, U_downarrow, U_uparrow, q, r, x, w, y):
    """Calculates rho using the simplified zeta (and gamma) function for direct value."""
    if not (L_downarrow < L_uparrow - 1e-9 and \
            U_downarrow < U_uparrow - 1e-9 and \
            L_uparrow <= U_downarrow + 1e-9):
        return 0.0

    g1 = gamma_func(L_downarrow, U_uparrow, q, x, w)
    g2 = gamma_func(L_downarrow, U_uparrow, r, w, y)
    g3 = gamma_func(L_uparrow, U_uparrow, q, x, w)   
    g4 = gamma_func(L_uparrow, U_uparrow, r, w, y)   
    g5 = gamma_func(L_downarrow, U_downarrow, q, x, w) 
    g6 = gamma_func(L_downarrow, U_downarrow, r, w, y) 
    g7 = gamma_func(L_uparrow, U_downarrow, q, x, w)   
    g8 = gamma_func(L_uparrow, U_downarrow, r, w, y)   

    rho_val = (g1 * g2) - (g3 * g4) - (g5 * g6) + (g7 * g8)
    return max(0.0, min(1.0, rho_val if math.isfinite(rho_val) else 0.0))

# --- Functions for Rejection Sampling (Section 5.1) ---

def pi_density_unnormalized(w, s, t, Xs, Xt):
    if t <= s + 1e-9: 
        return 1.0 if abs(w - Xs)<1e-9 and abs(s-t)<1e-9 else 0.0
        
    t_star = (s + t) / 2.0 
    l_duration = t - s
    q_duration = t_star - s
    r_duration = t - t_star

    if q_duration < -1e-9 or r_duration < -1e-9 : return 0.0
    if abs(q_duration) < 1e-9 : return 1.0 if abs(w - Xs) < 1e-9 else 0.0
    if abs(r_duration) < 1e-9 : return 1.0 if abs(w - Xt) < 1e-9 else 0.0

    mean_w = (r_duration / l_duration) * Xs + (q_duration / l_duration) * Xt
    variance_w = (q_duration * r_duration) / l_duration
    
    if variance_w < 1e-12: return 1.0 if abs(w - mean_w) < 1e-9 else 0.0

    exponent = -0.5 * ((w - mean_w)**2) / variance_w
    try: 
        res = math.exp(exponent)
        return res if math.isfinite(res) else 0.0
    except OverflowError: return 0.0 


def SZ_n_rho_bound(rho_params, n_for_SZ, get_upper_rho_bound):
    Ld, Lu, Ud, Uu, q, r, x, w, y = rho_params
    if not (Ld < Lu - 1e-9 and Ud < Uu - 1e-9 and Lu <= Ud + 1e-9):
        return 0.0

    k_for_zeta_terms = n_for_SZ + 1 
    zeta_params_list = [
        (Ld, Uu, q, x, w), (Ld, Uu, r, w, y), 
        (Lu, Uu, q, x, w), (Lu, Uu, r, w, y),  
        (Ld, Ud, q, x, w), (Ld, Ud, r, w, y),  
        (Lu, Ud, q, x, w), (Lu, Ud, r, w, y)   
    ]
    S_upper_for_zeta_i, S_lower_for_zeta_i = [], []
    for zp in zeta_params_list:
        s_u, s_l = get_alternating_zeta_sum_term(zp, k_for_zeta_terms)
        S_upper_for_zeta_i.append(s_u)
        S_lower_for_zeta_i.append(s_l)

    s1_U, s2_U, s3_U, s4_U, s5_U, s6_U, s7_U, s8_U = S_upper_for_zeta_i
    s1_L, s2_L, s3_L, s4_L, s5_L, s6_L, s7_L, s8_L = S_lower_for_zeta_i

    if get_upper_rho_bound: 
        term_A = (1.0 - s1_L - s2_L + s1_U * s2_U)
        term_B = (1.0 - s3_U - s4_U + s3_L * s4_L)
        term_C = (1.0 - s5_U - s6_U + s5_L * s6_L)
        term_D = (1.0 - s7_L - s8_L + s7_U * s8_U)
        sz_bound = term_A - term_B - term_C + term_D
    else: 
        term_A = (1.0 - s1_U - s2_U + s1_L * s2_L)
        term_B = (1.0 - s3_L - s4_L + s3_U * s4_U)
        term_C = (1.0 - s5_L - s6_L + s5_U * s6_U)
        term_D = (1.0 - s7_U - s8_U + s7_L * s8_L)
        sz_bound = term_A - term_B - term_C + term_D
    return max(0.0, min(1.0, sz_bound if math.isfinite(sz_bound) else 0.0))

def f1_integrand(u, Ld_rho_def, Lu_rho_def, Ud_rho_def, Uu_rho_def, 
                 q_dur, r_dur, Xs_bridge, Xt_bridge_end, 
                 s_pi_calc, t_pi_calc, n_for_SZ1_calc_integrand):
    pi_val = pi_density_unnormalized(u, s_pi_calc, t_pi_calc, Xs_bridge, Xt_bridge_end)
    if pi_val < 1e-12: return 0.0
    rho_params_for_SZ1 = (Ld_rho_def, Lu_rho_def, Ud_rho_def, Uu_rho_def, 
                          q_dur, r_dur, Xs_bridge, u, Xt_bridge_end)
    sz1_val = SZ_n_rho_bound(rho_params_for_SZ1, n_for_SZ1_calc_integrand, get_upper_rho_bound=True)
    val = sz1_val * pi_val
    if not math.isfinite(val):
        print(f"f1_integrand warning: non-finite val. u={u}, sz1_val={sz1_val}, pi_val={pi_val}")
    return val if math.isfinite(val) else 0.0

def evaluate_F1_cdf_numerical(v, L_integration_bound, U_integration_bound, f1_args_for_quad):
    if v <= L_integration_bound: return 0.0
    try:
        # Ensure integration limits are sensible
        low_b = min(L_integration_bound, v)
        high_b = max(L_integration_bound, v)
        if abs(high_b - low_b) < 1e-9 : return 0.0 # Interval is too small

        result, error = scipy.integrate.quad(f1_integrand, low_b, high_b, 
                                             args=f1_args_for_quad, epsabs=QUAD_EPS_ABS, 
                                             epsrel=QUAD_EPS_REL, limit=100) 
        
        # If v was less than L_integration_bound originally, integral should be negative of this result
        # However, CDF is F(v) = Int from overall_L_bound to v. So low_b should always be overall_L_bound.
        # The L_integration_bound here is the start of the domain for F1.
        
        return result if math.isfinite(result) else 0.0 
    except Exception as e: 
        print(f"quad integration error for F1_cdf(v={v:.3f}, L={L_integration_bound:.3f}): {type(e).__name__} {e}")
        return 0.0 # Fallback

def F1_cdf_minus_target_R(w_val_root, target_R_val_root, L_downarrow_root, U_uparrow_root, f1_args_root):
    current_cdf_val = evaluate_F1_cdf_numerical(w_val_root, L_downarrow_root, U_uparrow_root, f1_args_root)
    return current_cdf_val - target_R_val_root

def sample_from_f1_proposal_inverse_cdf(s, t, Xs, Xt, 
                                        L_rho_def_d, L_rho_def_u, 
                                        U_rho_def_d, U_rho_def_u,
                                        integration_domain_L, integration_domain_U, 
                                        n_for_SZ1_calc_sampling):
    q_duration = (t - s) / 2.0
    r_duration = (t - s) / 2.0
    f1_args_tuple = (L_rho_def_d, L_rho_def_u, U_rho_def_d, U_rho_def_u, 
                     q_duration, r_duration, Xs, Xt, s, t, n_for_SZ1_calc_sampling)

    if integration_domain_L >= integration_domain_U - 1e-9 : 
        return (integration_domain_L + integration_domain_U) / 2.0 if integration_domain_L <= integration_domain_U else integration_domain_L

    try:
        print(f"Norm_C: Integrating f1 from {integration_domain_L:.3f} to {integration_domain_U:.3f}")
        test_pts = [integration_domain_L, (integration_domain_L+integration_domain_U)/2, integration_domain_U]
        for pt in test_pts: print(f"f1_integrand({pt:.3f}) = {f1_integrand(pt, *f1_args_tuple):.3e}")

        norm_constant_C, norm_err = scipy.integrate.quad(f1_integrand, integration_domain_L, integration_domain_U, 
                                                         args=f1_args_tuple, epsabs=QUAD_EPS_ABS, epsrel=QUAD_EPS_REL, limit=100)
        if not math.isfinite(norm_constant_C) or norm_constant_C <= 1e-12:
            print(f"Warning: Norm C for F1 is non-finite/small: {norm_constant_C:.3e}. Range [{integration_domain_L:.3f}, {integration_domain_U:.3f}]. Args: {f1_args_tuple[:4]}")
            return random.uniform(integration_domain_L, integration_domain_U)
    except Exception as e:
        print(f"quad error for norm_C: {type(e).__name__} {e}")
        return random.uniform(integration_domain_L, integration_domain_U)

    R_uniform = random.random()
    R_target = R_uniform * norm_constant_C

    try:
        # The L_integration_bound for evaluate_F1_cdf_numerical is integration_domain_L
        val_at_a = F1_cdf_minus_target_R(integration_domain_L, R_target, integration_domain_L, integration_domain_U, f1_args_tuple)
        val_at_b = F1_cdf_minus_target_R(integration_domain_U, R_target, integration_domain_L, integration_domain_U, f1_args_tuple)
        
        if abs(val_at_a - val_at_b) < 1e-9 : # Function is flat, or R_target is outside range
             if abs(R_target) < 1e-9 * norm_constant_C : return integration_domain_L
             if abs(R_target - norm_constant_C) < 1e-9 * norm_constant_C : return integration_domain_U
             print(f"Warning: F1 is flat or R_target out of bounds for brentq. R_target={R_target:.3e}, F1(L)={val_at_a+R_target:.3e}, F1(U)={val_at_b+R_target:.3e}")
             return random.uniform(integration_domain_L, integration_domain_U)


        if val_at_a * val_at_b > 0: # Signs are the same
            # This means R_target is likely outside the [F1(L), F1(U)] range.
            # If R_target is very close to F1(L) (i.e., val_at_a is near 0), return L.
            # If R_target is very close to F1(U) (i.e., val_at_b is near 0), return U.
            if abs(val_at_a) < ROOT_FINDER_XTOL * 10 : return integration_domain_L
            if abs(val_at_b) < ROOT_FINDER_XTOL * 10 : return integration_domain_U
            print(f"Root finder sign issue: R_target={R_target:.3e}, F1(L)={val_at_a+R_target:.3e}, F1(U)={val_at_b+R_target:.3e}")
            print(f"val_at_a={val_at_a:.3e}, val_at_b={val_at_b:.3e}")
            return random.uniform(integration_domain_L, integration_domain_U) # Fallback

        w_star = scipy.optimize.brentq(F1_cdf_minus_target_R, integration_domain_L, integration_domain_U, 
                                       args=(R_target, integration_domain_L, integration_domain_U, f1_args_tuple),
                                       xtol=ROOT_FINDER_XTOL, rtol=ROOT_FINDER_XTOL, maxiter=100,
                                       disp=False) 
        return w_star
    except Exception as e: 
        print(f"Error in sample_from_f1_proposal_inverse_cdf (root finding): {type(e).__name__} {e}")
        return random.uniform(integration_domain_L, integration_domain_U)

def compare_rho_value_with_K_iterative(K_val, rho_params_iter):
    Ld, Lu, Ud, Uu, _, _, _, _, _ = rho_params_iter
    if not (Ld < Lu - 1e-9 and Ud < Uu - 1e-9 and Lu <= Ud + 1e-9): return False
    for n_iter in range(MAX_ITER_COMPARISON):
        rho_lower_bound = SZ_n_rho_bound(rho_params_iter, n_iter, get_upper_rho_bound=False)
        rho_upper_bound = SZ_n_rho_bound(rho_params_iter, n_iter, get_upper_rho_bound=True)
        if K_val < rho_lower_bound - 1e-12: return True  
        if K_val > rho_upper_bound + 1e-12: return False 
        if abs(rho_upper_bound - rho_lower_bound) < 1e-12: return K_val < rho_lower_bound - 1e-15 
    final_rho_lower = SZ_n_rho_bound(rho_params_iter, MAX_ITER_COMPARISON -1, get_upper_rho_bound=False)
    return K_val < final_rho_lower - 1e-15

# --- Main Sampling Function (Section 5.1) ---
def sample_midpoint_conditional(s, t, Xs, Xt, 
                                L_initial_downarrow, L_initial_uparrow, 
                                U_initial_downarrow, U_initial_uparrow):
    q = (t - s) / 2.0  
    r = (t - s) / 2.0  

    if q < 1e-9: 
        w_star_val = Xs 
        adj_L_downarrow = L_initial_downarrow
        adj_L_uparrow   = min(L_initial_uparrow, w_star_val) if L_initial_uparrow is not None else w_star_val
        adj_U_downarrow = max(U_initial_downarrow, w_star_val) if U_initial_downarrow is not None else w_star_val
        adj_U_uparrow   = U_initial_uparrow
        if L_initial_uparrow is None: adj_L_uparrow = w_star_val 
        if U_initial_downarrow is None: adj_U_downarrow = w_star_val
        if adj_L_downarrow is not None and adj_L_uparrow is not None and adj_L_downarrow > adj_L_uparrow: adj_L_uparrow = adj_L_downarrow 
        if adj_U_downarrow is not None and adj_U_uparrow is not None and adj_U_downarrow > adj_U_uparrow: adj_U_uparrow = adj_U_downarrow 
        if adj_L_uparrow is not None and adj_U_downarrow is not None and adj_L_uparrow > adj_U_downarrow + 1e-9:
            adj_L_uparrow = w_star_val
            adj_U_downarrow = w_star_val
        return w_star_val, adj_L_downarrow, adj_L_uparrow, adj_U_downarrow, adj_U_uparrow

    n_for_SZ1_calc = 0 
    max_rejection_attempts = 1000 
    
    w_star_domain_L = L_initial_downarrow
    w_star_domain_U = U_initial_uparrow
    if w_star_domain_L is None or w_star_domain_U is None or w_star_domain_L >= w_star_domain_U - 1e-9:
        mid_mean_uncond = (Xs+Xt)/2.0
        std_dev_uncond = math.sqrt((t-s)/4 if (t-s)>0 else 0.01)
        w_star_domain_L = L_initial_downarrow if L_initial_downarrow is not None else mid_mean_uncond - 5 * std_dev_uncond
        w_star_domain_U = U_initial_uparrow if U_initial_uparrow is not None else mid_mean_uncond + 5 * std_dev_uncond
        if w_star_domain_L >= w_star_domain_U - 1e-9: 
            w_star_domain_L = mid_mean_uncond - 1e-3
            w_star_domain_U = mid_mean_uncond + 1e-3

    for attempt in range(max_rejection_attempts):
        w_star = sample_from_f1_proposal_inverse_cdf(s, t, Xs, Xt, 
                                                     L_initial_downarrow, L_initial_uparrow, 
                                                     U_initial_downarrow, U_initial_uparrow,
                                                     w_star_domain_L, w_star_domain_U,     
                                                     n_for_SZ1_calc)
        pi_w_star = pi_density_unnormalized(w_star, s, t, Xs, Xt)
        if pi_w_star < 1e-12: continue
        
        L_adj_downarrow = L_initial_downarrow
        L_adj_uparrow   = min(L_initial_uparrow, w_star) if L_initial_uparrow is not None else w_star
        U_adj_downarrow = max(U_initial_downarrow, w_star) if U_initial_downarrow is not None else w_star
        U_adj_uparrow   = U_initial_uparrow
        if L_initial_uparrow is None: L_adj_uparrow = w_star 
        if U_initial_downarrow is None: U_adj_downarrow = w_star 
        
        if not (L_adj_downarrow is not None and L_adj_uparrow is not None and \
                U_adj_downarrow is not None and U_adj_uparrow is not None and \
                L_adj_downarrow < L_adj_uparrow - 1e-9 and \
                U_adj_downarrow < U_adj_uparrow - 1e-9 and \
                L_adj_uparrow <= U_adj_downarrow + 1e-9):
            continue
        rho_params_for_comparison = (L_adj_downarrow, L_adj_uparrow, U_adj_downarrow, U_adj_uparrow, 
                                     q, r, Xs, w_star, Xt)
        sz1_for_acceptance = SZ_n_rho_bound(rho_params_for_comparison, n_for_SZ=n_for_SZ1_calc, get_upper_rho_bound=True)
        if sz1_for_acceptance < 1e-12: continue
        R_uniform = random.random()
        K_val = R_uniform * sz1_for_acceptance 
        accept_condition_met = compare_rho_value_with_K_iterative(K_val, rho_params_for_comparison)
        if accept_condition_met:
            return w_star, L_adj_downarrow, L_adj_uparrow, U_adj_downarrow, U_adj_uparrow
    return (Xs + Xt) / 2.0, L_initial_downarrow, L_initial_uparrow, U_initial_downarrow, U_initial_uparrow

# --- Step 2 of Bisection: Updating Layers Given X_t* (Section 5.2) ---
def update_layers_after_midpoint(s, t, Xs, X_mid, Xt, 
                                 L_glob_adj_downarrow, L_glob_adj_uparrow,
                                 U_glob_adj_downarrow, U_glob_adj_uparrow):
    t_star = (s + t) / 2.0
    q = t_star - s 
    r = t - t_star   
    if q < 1e-9 or r < 1e-9: 
        L_s_tstar_d, L_s_tstar_u = min(Xs, X_mid), min(Xs, X_mid)
        U_s_tstar_d, U_s_tstar_u = max(Xs, X_mid), max(Xs, X_mid)
        L_tstar_t_d, L_tstar_t_u = min(X_mid, Xt), min(X_mid, Xt)
        U_tstar_t_d, U_tstar_t_u = max(X_mid, Xt), max(X_mid, Xt)
        return ([L_s_tstar_d, L_s_tstar_u], [U_s_tstar_d, U_s_tstar_u]), \
               ([L_tstar_t_d, L_tstar_t_u], [U_tstar_t_d, U_tstar_t_u])

    min_Xs_Xmid, max_Xs_Xmid = min(Xs, X_mid), max(Xs, X_mid)
    min_Xmid_Xt, max_Xmid_Xt = min(X_mid, Xt), max(X_mid, Xt)
    
    if not (L_glob_adj_downarrow is not None and L_glob_adj_uparrow is not None and \
            U_glob_adj_downarrow is not None and U_glob_adj_uparrow is not None and \
            L_glob_adj_downarrow < L_glob_adj_uparrow - 1e-9 and \
            U_glob_adj_downarrow < U_glob_adj_uparrow - 1e-9 and \
            L_glob_adj_uparrow <= U_glob_adj_downarrow + 1e-9):
        chosen_scenario_idx = 0 
        L_s_d_fb, L_s_u_fb = min(Xs,X_mid), min(Xs,X_mid)
        U_s_d_fb, U_s_u_fb = max(Xs,X_mid), max(Xs,X_mid)
        L_r_d_fb, L_r_u_fb = min(X_mid,Xt), min(X_mid,Xt)
        U_r_d_fb, U_r_u_fb = max(X_mid,Xt), max(X_mid,Xt)
        return ([L_s_d_fb, L_s_u_fb], [U_s_d_fb, U_s_u_fb]), ([L_r_d_fb, L_r_u_fb], [U_r_d_fb, U_r_u_fb])

    probs_numerator = [0.0] * 9
    probs_numerator[0] = beta_func(L_glob_adj_downarrow, L_glob_adj_uparrow, U_glob_adj_downarrow, U_glob_adj_uparrow, q, Xs, X_mid) * \
                         beta_func(L_glob_adj_downarrow, L_glob_adj_uparrow, U_glob_adj_downarrow, U_glob_adj_uparrow, r, X_mid, Xt)
    probs_numerator[1] = beta_func(L_glob_adj_downarrow, L_glob_adj_uparrow, U_glob_adj_downarrow, U_glob_adj_uparrow, q, Xs, X_mid) * \
                         beta_func(L_glob_adj_uparrow, min_Xmid_Xt, U_glob_adj_downarrow, U_glob_adj_uparrow, r, X_mid, Xt)
    probs_numerator[2] = beta_func(L_glob_adj_downarrow, L_glob_adj_uparrow, U_glob_adj_downarrow, U_glob_adj_uparrow, q, Xs, X_mid) * \
                         beta_func(L_glob_adj_downarrow, L_glob_adj_uparrow, max_Xmid_Xt, U_glob_adj_downarrow, r, X_mid, Xt)
    probs_numerator[3] = beta_func(L_glob_adj_downarrow, L_glob_adj_uparrow, U_glob_adj_downarrow, U_glob_adj_uparrow, q, Xs, X_mid) * \
                         beta_func(L_glob_adj_uparrow, min_Xmid_Xt, max_Xmid_Xt, U_glob_adj_downarrow, r, X_mid, Xt)
    probs_numerator[4] = beta_func(L_glob_adj_uparrow, min_Xs_Xmid, U_glob_adj_downarrow, U_glob_adj_uparrow, q, Xs, X_mid) * \
                         beta_func(L_glob_adj_downarrow, L_glob_adj_uparrow, U_glob_adj_downarrow, U_glob_adj_uparrow, r, X_mid, Xt)
    probs_numerator[5] = beta_func(L_glob_adj_uparrow, min_Xs_Xmid, U_glob_adj_downarrow, U_glob_adj_uparrow, q, Xs, X_mid) * \
                         beta_func(L_glob_adj_uparrow, min_Xmid_Xt, U_glob_adj_downarrow, U_glob_adj_uparrow, r, X_mid, Xt)
    probs_numerator[6] = beta_func(L_glob_adj_downarrow, L_glob_adj_uparrow, max_Xs_Xmid, U_glob_adj_downarrow, q, Xs, X_mid) * \
                         beta_func(L_glob_adj_downarrow, L_glob_adj_uparrow, U_glob_adj_downarrow, U_glob_adj_uparrow, r, X_mid, Xt)
    probs_numerator[7] = beta_func(L_glob_adj_downarrow, L_glob_adj_uparrow, max_Xs_Xmid, U_glob_adj_downarrow, q, Xs, X_mid) * \
                         beta_func(L_glob_adj_uparrow, min_Xmid_Xt, U_glob_adj_downarrow, U_glob_adj_uparrow, r, X_mid, Xt)
    probs_numerator[8] = beta_func(L_glob_adj_uparrow, min_Xs_Xmid, max_Xs_Xmid, U_glob_adj_downarrow, q, Xs, X_mid) * \
                         beta_func(L_glob_adj_downarrow, L_glob_adj_uparrow, U_glob_adj_downarrow, U_glob_adj_uparrow, r, X_mid, Xt)
    
    sum_probs_numerator = sum(p for p in probs_numerator if math.isfinite(p) and p >= 0)
    if sum_probs_numerator < 1e-12: chosen_scenario_idx = 0 
    else:
        normalized_probs = [p / sum_probs_numerator if math.isfinite(p) and p >=0 else 0.0 for p in probs_numerator]
        try: chosen_scenario_idx = random.choices(range(9), weights=normalized_probs, k=1)[0]
        except ValueError: chosen_scenario_idx = 0

    L_s_d, L_s_u = L_glob_adj_downarrow, L_glob_adj_uparrow
    U_s_d, U_s_u = U_glob_adj_downarrow, U_glob_adj_uparrow
    L_r_d, L_r_u = L_glob_adj_downarrow, L_glob_adj_uparrow
    U_r_d, U_r_u = U_glob_adj_downarrow, U_glob_adj_uparrow
    scenario_indicators = [(1,1,1,1), (1,1,0,1), (1,1,1,0), (1,1,0,0), (0,1,1,1), (0,1,0,1), (1,0,1,1), (1,0,0,1), (0,0,1,1)]
    m_L_ind, M_L_ind, m_R_ind, M_R_ind = scenario_indicators[chosen_scenario_idx]
    if m_L_ind == 0: L_s_d, L_s_u = L_glob_adj_uparrow, min_Xs_Xmid
    if M_L_ind == 0: U_s_d, U_s_u = max_Xs_Xmid, U_glob_adj_downarrow
    if m_R_ind == 0: L_r_d, L_r_u = L_glob_adj_uparrow, min_Xmid_Xt
    if M_R_ind == 0: U_r_d, U_r_u = max_Xmid_Xt, U_glob_adj_downarrow

    def sanitize_layer_final(Ld, Lu, Ud, Uu, endpoint1, endpoint2):
        # Ensure basic order Ld <= Lu and Ud <= Uu from scenario
        Ld_final, Lu_final = min(Ld, Lu), max(Ld, Lu)
        Ud_final, Uu_final = min(Ud, Uu), max(Ud, Uu)
        # Ensure intervals are valid (low < high)
        if Ld_final >= Lu_final - 1e-9: Lu_final = Ld_final + 1e-9
        if Ud_final >= Uu_final - 1e-9: Uu_final = Ud_final + 1e-9
        # Ensure min_interval_max <= max_interval_min
        if Lu_final > Ud_final + 1e-9: # Problematic overlap
            midpoint = (Lu_final + Ud_final) / 2.0
            Lu_final, Ud_final = midpoint, midpoint
            if Ld_final >= Lu_final - 1e-9: Ld_final = Lu_final - 1e-9
            if Uu_final <= Ud_final + 1e-9: Uu_final = Ud_final + 1e-9
        return Ld_final, Lu_final, Ud_final, Uu_final

    L_s_d, L_s_u, U_s_d, U_s_u = sanitize_layer_final(L_s_d, L_s_u, U_s_d, U_s_u, Xs, X_mid)
    L_r_d, L_r_u, U_r_d, U_r_u = sanitize_layer_final(L_r_d, L_r_u, U_r_d, U_r_u, X_mid, Xt)
    return ([L_s_d, L_s_u], [U_s_d, U_s_u]), ([L_r_d, L_r_u], [U_r_d, U_r_u])

# --- Bisection Procedure (Table 1) ---
def bisect_intersection_layer(s, t, Xs, Xt, 
                              L_initial_downarrow, L_initial_uparrow, 
                              U_initial_downarrow, U_initial_uparrow):
    if abs(t - s) < 1e-9: return None, None
    t_star = (s + t) / 2.0
    X_mid, L_glob_adj_d, L_glob_adj_u, U_glob_adj_d, U_glob_adj_u = \
        sample_midpoint_conditional(s, t, Xs, Xt, L_initial_downarrow, L_initial_uparrow, U_initial_downarrow, U_initial_uparrow)
    layers_L_bridge, layers_R_bridge = \
        update_layers_after_midpoint(s, t, Xs, X_mid, Xt, L_glob_adj_d, L_glob_adj_u, U_glob_adj_d, U_glob_adj_u)
    L_left_min_interval, U_left_max_interval = layers_L_bridge
    L_right_min_interval, U_right_max_interval = layers_R_bridge
    layer_left_info = (s, t_star, Xs, X_mid, L_left_min_interval[0], L_left_min_interval[1], U_left_max_interval[0], U_left_max_interval[1])    
    layer_right_info = (t_star, t, X_mid, Xt, L_right_min_interval[0], L_right_min_interval[1], U_right_max_interval[0], U_right_max_interval[1]) 
    return layer_left_info, layer_right_info

# --- Refinement Procedure (Section 2.1, 5.3) ---
def SZ_n_beta_bound(beta_params_tuple, n_for_SZ, get_upper_beta_bound):
    """
    Calculates alternating series bounds SZ_{2n} or SZ_{2n+1} for a single beta function.
    beta_params_tuple: (L_low, L_high, U_low, U_high, l_dur, x_start, y_end)
    n_for_SZ: 'n' index for SZ_{2n} or SZ_{2n+1}.
    get_upper_beta_bound: True for SZ_{2n+1}, False for SZ_{2n}.
    
    beta = zeta_B + zeta_C - zeta_A - zeta_D (using A,B,C,D for the four zeta terms)
    zeta_A = zeta(L_low, U_high), zeta_B = zeta(L_high, U_high)
    zeta_C = zeta(L_low, U_low),  zeta_D = zeta(L_high, U_low)
    (Based on beta = g1-g2-g3+g4 = (1-zA)-(1-zB)-(1-zC)+(1-zD) = zB+zC-zA-zD)
    """
    L_low, L_high, U_low, U_high, l_dur, x, y = beta_params_tuple
    if not (L_low < L_high - 1e-9 and U_low < U_high - 1e-9 and L_high <= U_low + 1e-9):
        return 0.0 # Beta is 0 if intervals are invalid

    k_for_zeta_terms = n_for_SZ + 1

    params_zA = (L_low, U_high, l_dur, x, y)
    params_zB = (L_high, U_high, l_dur, x, y)
    params_zC = (L_low, U_low, l_dur, x, y)
    params_zD = (L_high, U_low, l_dur, x, y)

    sU_zA, sL_zA = get_alternating_zeta_sum_term(params_zA, k_for_zeta_terms)
    sU_zB, sL_zB = get_alternating_zeta_sum_term(params_zB, k_for_zeta_terms)
    sU_zC, sL_zC = get_alternating_zeta_sum_term(params_zC, k_for_zeta_terms)
    sU_zD, sL_zD = get_alternating_zeta_sum_term(params_zD, k_for_zeta_terms)

    if get_upper_beta_bound: # SZ_{2n+1} for beta
        # For beta = zB + zC - zA - zD
        # Upper bound: Use Upper for zB, zC; Lower for zA, zD
        sz_bound = sU_zB + sU_zC - sL_zA - sL_zD
    else: # SZ_{2n} for beta
        # Lower bound: Use Lower for zB, zC; Upper for zA, zD
        sz_bound = sL_zB + sL_zC - sU_zA - sU_zD
        
    return max(0.0, min(1.0, sz_bound if math.isfinite(sz_bound) else 0.0))


def compare_beta_value_with_K_iterative(K_val, beta_params_iter):
    """
    Compares K_val with beta(beta_params_iter) using alternating series for beta.
    Returns True if K_val < beta, False if K_val >= beta.
    """
    L_low, L_high, U_low, U_high, _, _, _ = beta_params_iter
    if not (L_low < L_high - 1e-9 and U_low < U_high - 1e-9 and L_high <= U_low + 1e-9):
        return False # Beta is 0, so K < 0 is false

    for n_iter in range(MAX_ITER_COMPARISON):
        beta_lower_bound = SZ_n_beta_bound(beta_params_iter, n_iter, get_upper_beta_bound=False)
        beta_upper_bound = SZ_n_beta_bound(beta_params_iter, n_iter, get_upper_beta_bound=True)

        if K_val < beta_lower_bound - 1e-12: return True  
        if K_val > beta_upper_bound + 1e-12: return False 
        if abs(beta_upper_bound - beta_lower_bound) < 1e-12: 
            return K_val < beta_lower_bound - 1e-15 
            
    final_beta_lower = SZ_n_beta_bound(beta_params_iter, MAX_ITER_COMPARISON -1, get_upper_beta_bound=False)
    return K_val < final_beta_lower - 1e-15


def refine_intersection_layer(s, t, Xs, Xt, 
                              L_current_d, L_current_u, 
                              U_current_d, U_current_u, 
                              refine_min_or_max):
    """
    Implements the Refine procedure (Section 2.1, 5.3).
    Args:
        refine_min_or_max (str): "min" or "max" to indicate which extremum to refine.
    Returns:
        Updated (L_new_d, L_new_u, U_new_d, U_new_u)
    """
    l_duration = t - s
    if l_duration < 1e-9: # Cannot refine zero-duration interval layers this way
        return L_current_d, L_current_u, U_current_d, U_current_u

    L_new_d, L_new_u = L_current_d, L_current_u
    U_new_d, U_new_u = U_current_d, U_current_u

    if refine_min_or_max == "min":
        if abs(L_current_u - L_current_d) < 1e-9: # Interval already too small
            return L_current_d, L_current_u, U_current_d, U_current_u
        L_mid = (L_current_d + L_current_u) / 2.0
        if abs(L_mid - L_current_d) < 1e-9 or abs(L_mid - L_current_u) < 1e-9: # Midpoint too close to bounds
            return L_current_d, L_current_u, U_current_d, U_current_u

        # Prob that m is in the upper half [L_mid, L_current_u]
        # P_target = beta_num / beta_den
        # beta_num_params = (L_mid, L_current_u, U_current_d, U_current_u, l_duration, Xs, Xt)
        # beta_den_params = (L_current_d, L_current_u, U_current_d, U_current_u, l_duration, Xs, Xt)
        
        # Check validity of intervals for beta
        # For beta_num: min in [L_mid, L_current_u], max in [U_current_d, U_current_u]
        # Requires L_mid < L_current_u AND U_current_d < U_current_u AND L_current_u <= U_current_d
        beta_num_params_valid = (L_mid < L_current_u - 1e-9 and \
                                 U_current_d < U_current_u - 1e-9 and \
                                 L_current_u <= U_current_d + 1e-9)
        
        # For beta_den: min in [L_current_d, L_current_u], max in [U_current_d, U_current_u]
        beta_den_params_valid = (L_current_d < L_current_u - 1e-9 and \
                                 U_current_d < U_current_u - 1e-9 and \
                                 L_current_u <= U_current_d + 1e-9)


        val_beta_den = beta_func(L_current_d, L_current_u, U_current_d, U_current_u, l_duration, Xs, Xt) if beta_den_params_valid else 0.0

        if val_beta_den < 1e-12: # Denominator is zero, cannot refine based on this probability
            return L_current_d, L_current_u, U_current_d, U_current_u

        K_val = random.random() * val_beta_den
        
        # Parameters for beta_num, which is P(m in [L_mid, L_current_u] AND M in [U_current_d, U_current_u])
        beta_num_params = (L_mid, L_current_u, U_current_d, U_current_u, l_duration, Xs, Xt)
        m_is_in_upper_half = compare_beta_value_with_K_iterative(K_val, beta_num_params) if beta_num_params_valid else False

        if m_is_in_upper_half:
            L_new_d = L_mid
            # L_new_u remains L_current_u
        else:
            # L_new_d remains L_current_d
            L_new_u = L_mid
        
    elif refine_min_or_max == "max":
        if abs(U_current_u - U_current_d) < 1e-9: # Interval already too small
            return L_current_d, L_current_u, U_current_d, U_current_u
        U_mid = (U_current_d + U_current_u) / 2.0
        if abs(U_mid - U_current_d) < 1e-9 or abs(U_mid - U_current_u) < 1e-9: # Midpoint too close to bounds
            return L_current_d, L_current_u, U_current_d, U_current_u

        # Prob that M is in the upper half [U_mid, U_current_u]
        # P_target = beta_num / beta_den
        # beta_num_params = (L_current_d, L_current_u, U_mid, U_current_u, l_duration, Xs, Xt)
        # beta_den_params = (L_current_d, L_current_u, U_current_d, U_current_u, l_duration, Xs, Xt)
        
        # For beta_num: min in [L_current_d, L_current_u], max in [U_mid, U_current_u]
        # Requires L_current_d < L_current_u AND U_mid < U_current_u AND L_current_u <= U_mid
        beta_num_params_valid = (L_current_d < L_current_u - 1e-9 and \
                                 U_mid < U_current_u - 1e-9 and \
                                 L_current_u <= U_mid + 1e-9)

        # For beta_den: min in [L_current_d, L_current_u], max in [U_current_d, U_current_u]
        beta_den_params_valid = (L_current_d < L_current_u - 1e-9 and \
                                 U_current_d < U_current_u - 1e-9 and \
                                 L_current_u <= U_current_d + 1e-9)

        val_beta_den = beta_func(L_current_d, L_current_u, U_current_d, U_current_u, l_duration, Xs, Xt) if beta_den_params_valid else 0.0

        if val_beta_den < 1e-12:
            return L_current_d, L_current_u, U_current_d, U_current_u

        K_val = random.random() * val_beta_den
        
        beta_num_params = (L_current_d, L_current_u, U_mid, U_current_u, l_duration, Xs, Xt)
        M_is_in_upper_half = compare_beta_value_with_K_iterative(K_val, beta_num_params) if beta_num_params_valid else False

        if M_is_in_upper_half:
            U_new_d = U_mid
            # U_new_u remains U_current_u
        else:
            # U_new_d remains U_current_d
            U_new_u = U_mid
    else:
        raise ValueError("refine_min_or_max must be 'min' or 'max'")

    return L_new_d, L_new_u, U_new_d, U_new_u

# --- Epsilon-Strong Algorithm (Table 2) ---
def epsilon_strong_algorithm(X0, X1, initial_L_d, initial_L_u, initial_U_d, initial_U_u, num_iterations_n):
    """
    Implements the epsilon-strong algorithm from Table 2 of the paper.
    
    Args:
        X0, X1: Start and end values of the Brownian bridge on [0,1].
        initial_L_d, initial_L_u: Initial bounds for the minimum of the bridge on [0,1].
        initial_U_d, initial_U_u: Initial bounds for the maximum of the bridge on [0,1].
        num_iterations_n: The number of bisection iterations 'n' from the paper.
        
    Returns:
        P_layers: A list of the final $2^n$ intersection layers.
                  Each layer is a tuple: (s, t, Xs, Xt, Ld, Lu, Ud, Uu)
    """
    # 1. Initialize P = {I_0,1}
    # Layer format: (s, t, Xs, Xt, L_downarrow, L_uparrow, U_downarrow, U_uparrow)
    current_layer_info = (0.0, 1.0, X0, X1, initial_L_d, initial_L_u, initial_U_d, initial_U_u)
    P_layers = [current_layer_info]

    # Iteration loop (i from 1 to num_iterations_n in paper's notation)
    for i_iter in range(num_iterations_n):
        print(f"\nStarting Epsilon-Strong Iteration {i_iter + 1}/{num_iterations_n}")
        next_P_layers = []
        for layer_s_t in P_layers:
            s_curr, t_curr, Xs_curr, Xt_curr, Ld_curr, Lu_curr, Ud_curr, Uu_curr = layer_s_t
            
            # 2.i. Bisect the information I_s,t into I_s,t* and I_t*,t
            #      where t* = (s+t)/2
            if abs(t_curr - s_curr) < 1e-9: # Interval too small, cannot bisect further
                next_P_layers.append(layer_s_t) # Keep the current layer
                continue

            layer_left_bisection, layer_right_bisection = bisect_intersection_layer(
                s_curr, t_curr, Xs_curr, Xt_curr, Ld_curr, Lu_curr, Ud_curr, Uu_curr
            )

            if layer_left_bisection is None or layer_right_bisection is None:
                # Bisection failed (e.g. interval was already zero), keep original layer
                # This case should be caught by the abs(t_curr - s_curr) check ideally
                next_P_layers.append(layer_s_t)
                continue

            # Unpack the two new layers from bisection
            s_L, t_L, Xs_L, Xt_L, Ld_L_bis, Lu_L_bis, Ud_L_bis, Uu_L_bis = layer_left_bisection
            s_R, t_R, Xs_R, Xt_R, Ld_R_bis, Lu_R_bis, Ud_R_bis, Uu_R_bis = layer_right_bisection
            
            # 2.ii. Refine I_s,t* and I_t*,t until width of layers <= sqrt((t-s)/2)
            #       The (t-s)/2 here refers to the length of the *new* interval.
            
            new_layers_after_refinement = []
            for new_layer_from_bisection in [layer_left_bisection, layer_right_bisection]:
                s_new, t_new, Xs_new, Xt_new, Ld_new, Lu_new, Ud_new, Uu_new = new_layer_from_bisection
                
                interval_length_new = t_new - s_new
                if interval_length_new < 1e-9: # No refinement if interval is zero length
                    new_layers_after_refinement.append(new_layer_from_bisection)
                    continue

                target_width = math.sqrt(interval_length_new) # Paper: sqrt((t-s)/2) which is sqrt(new_interval_length)
                
                # Refine min layer
                for _refine_attempt_min in range(MAX_REFINE_ITER): # Max attempts to prevent infinite loop
                    min_layer_width = Lu_new - Ld_new
                    if min_layer_width <= target_width + 1e-9: # Add tolerance
                        break
                    Ld_new, Lu_new, Ud_new, Uu_new = refine_intersection_layer(
                        s_new, t_new, Xs_new, Xt_new, Ld_new, Lu_new, Ud_new, Uu_new, "min"
                    )
                
                # Refine max layer
                for _refine_attempt_max in range(MAX_REFINE_ITER): # Max attempts
                    max_layer_width = Uu_new - Ud_new
                    if max_layer_width <= target_width + 1e-9: # Add tolerance
                        break
                    Ld_new, Lu_new, Ud_new, Uu_new = refine_intersection_layer(
                        s_new, t_new, Xs_new, Xt_new, Ld_new, Lu_new, Ud_new, Uu_new, "max"
                    )
                
                refined_layer_tuple = (s_new, t_new, Xs_new, Xt_new, Ld_new, Lu_new, Ud_new, Uu_new)
                new_layers_after_refinement.append(refined_layer_tuple)
            
            next_P_layers.extend(new_layers_after_refinement)
            
        # 3. Collect updated information
        P_layers = next_P_layers
        print(f"Iteration {i_iter + 1}: Number of layers = {len(P_layers)}")

    # 4. Return P (the collection of final layers)
    return P_layers




def plot_epsilon_strong_layers(final_layers, initial_X0, initial_X1, title="Epsilon-Strong Algorithm Layers"):
    """
    Plots the results of the epsilon-strong algorithm.

    Args:
        final_layers (list): A list of the final $2^n$ intersection layers.
                             Each layer is a tuple: (s, t, Xs, Xt, Ld, Lu, Ud, Uu)
        initial_X0 (float): The starting value of the overall bridge at t=0.
        initial_X1 (float): The ending value of the overall bridge at t=1 (or final time).
        title (str): The title for the plot.
    """
    if not final_layers:
        print("No layers to plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    # Collect all unique time points and corresponding X values for the main path
    path_points_t = {final_layers[0][0]} # Start time of the first segment
    path_points_x = {final_layers[0][2]} # Xs of the first segment

    for layer in final_layers:
        s_f, t_f, Xs_f, Xt_f, Ld_f, Lu_f, Ud_f, Uu_f = layer
        
        # Add points for the main path (connecting Xs_f, Xt_f)
        path_points_t.add(s_f)
        path_points_t.add(t_f)
        path_points_x.add(Xs_f) # X value at s_f
        path_points_x.add(Xt_f) # X value at t_f

        # Plot min/max layer bounds for this segment
        # Min layer: Ld_f to Lu_f. We are interested in the overall lower bound Ld_f.
        # Max layer: Ud_f to Uu_f. We are interested in the overall upper bound Uu_f.
        
        # Lower bound for the path in this segment (L_downarrow from the min layer)
        ax.plot([s_f, t_f], [Ld_f, Ld_f], color='blue', linestyle='-', alpha=0.7, linewidth=1.5)
        # Upper bound for the path in this segment (U_uparrow from the max layer)
        ax.plot([s_f, t_f], [Uu_f, Uu_f], color='red', linestyle='-', alpha=0.7, linewidth=1.5)

        # Shade the allowed region for extrema (optional, can make plot busy)
        # ax.fill_between([s_f, t_f], Ld_f, Lu_f, color='blue', alpha=0.1, label='Min Layer Range' if idx==0 else "")
        # ax.fill_between([s_f, t_f], Ud_f, Uu_f, color='red', alpha=0.1, label='Max Layer Range' if idx==0 else "")


    # Sort path points by time to plot the main bridge path correctly
    # We need to reconstruct the sequence of X values at the bisection points
    # The final_layers list gives segments. We need to order them.
    
    # Create a dictionary to map start times to X values and end times
    # This helps in reconstructing the path if layers are not perfectly ordered by time
    # (though they should be if P_layers is processed sequentially)
    
    # A simpler way: plot each segment's Xs -> Xt
    all_t_coords = []
    all_x_coords = []

    # Ensure layers are sorted by start time for correct path plotting
    sorted_layers = sorted(final_layers, key=lambda l: l[0])

    if sorted_layers:
        all_t_coords.append(sorted_layers[0][0]) # s of first segment
        all_x_coords.append(sorted_layers[0][2]) # Xs of first segment
        for layer in sorted_layers:
            all_t_coords.append(layer[1]) # t of current segment
            all_x_coords.append(layer[3]) # Xt of current segment
    
    # Plot the sampled Brownian bridge path (connecting the X_mid points)
    ax.plot(all_t_coords, all_x_coords, 'ko-', label='Sampled Path Points', markersize=4, linewidth=1)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value (X_t)")
    ax.set_title(title)
    
    # Create custom legend entries for bounds
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='k', marker='o', linestyle='-', label='Sampled Path Points'),
                       Line2D([0], [0], color='blue', linestyle='-', label='Lower Bound for Path (Ld)'),
                       Line2D([0], [0], color='red', linestyle='-', label='Upper Bound for Path (Uu)')]
    ax.legend(handles=legend_elements, loc='best')
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Set y-limits to be a bit wider than the observed range to see bounds clearly
    min_val_overall = min(layer[4] for layer in final_layers) # min Ld
    max_val_overall = max(layer[7] for layer in final_layers) # max Uu
    min_x_path = min(all_x_coords) if all_x_coords else initial_X0
    max_x_path = max(all_x_coords) if all_x_coords else initial_X1

    plot_min_y = min(min_val_overall, min_x_path)
    plot_max_y = max(max_val_overall, max_x_path)
    padding = (plot_max_y - plot_min_y) * 0.1 if (plot_max_y - plot_min_y) > 1e-6 else 0.1
    ax.set_ylim(plot_min_y - padding, plot_max_y + padding)
    
    # Ensure x-axis covers [0,1] or the full time span of the layers
    ax.set_xlim(sorted_layers[0][0] if sorted_layers else 0, sorted_layers[-1][1] if sorted_layers else 1)

    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    s_time = 0.0
    t_time = 1.0
    X_s_val = 0.0
    X_t_val = 0.1 

    # Initial layer for the whole interval [s,t]
    L0_d = -0.5  
    L0_u = 0.0   # Min of path is in [-0.5, 0.0]
    U0_d = 0.05  # Max of path is in [0.05, 0.6]
    U0_u = 0.6   
    
    valid_layer = True
    if not (L0_d is not None and L0_u is not None and U0_d is not None and U0_u is not None and \
            L0_d < L0_u - 1e-9 and U0_d < U0_u - 1e-9 and L0_u <= U0_d + 1e-9):  #L_high <= U_low
        print(f"Initial Layer definition is inconsistent: L:[{L0_d}, {L0_u}], U:[{U0_d}, {U0_u}].")
        valid_layer = False
    if not (L0_d - 1e-9 <= X_s_val <= U0_u + 1e-9 and L0_d - 1e-9 <= X_t_val <= U0_u + 1e-9): 
        print(f"Start/End points Xs={X_s_val}, Xt={X_t_val} are outside the overall initial layer [{L0_d}, {U0_u}].")
        valid_layer = False
    
    if valid_layer:
        print(f"--- Testing Epsilon-Strong Algorithm ---")
        print(f"Initial Bridge: from ({s_time},{X_s_val}) to ({t_time},{X_t_val})")
        print(f"Initial Layer I_0: Min in [{L0_d:.3f}, {L0_u:.3f}], Max in [{U0_d:.3f}, {U0_u:.3f}]")

        num_bisection_iterations = 3 # Example: 3 iterations will result in 2^3 = 8 layers
        
        final_layers = epsilon_strong_algorithm(X_s_val, X_t_val, L0_d, L0_u, U0_d, U0_u, num_bisection_iterations)
        
        print(f"\nEpsilon-Strong Algorithm completed after {num_bisection_iterations} iterations.")
        print(f"Total number of final layers: {len(final_layers)}")
        
        plot_epsilon_strong_layers(final_layers, X_s_val, X_t_val, 
                               title=f"Epsilon-Strong Layers after {num_bisection_iterations} iterations")

        # for idx, layer in enumerate(final_layers):
        #     s_f, t_f, Xs_f, Xt_f, Ld_f, Lu_f, Ud_f, Uu_f = layer
        #     print(f"\nLayer {idx+1}:")
        #     print(f"  Interval: [{s_f:.3f}, {t_f:.3f}], Bridge: {Xs_f:.3f} -> {Xt_f:.3f}")
        #     print(f"  Min Layer: [{Ld_f:.3f}, {Lu_f:.3f}], Max Layer: [{Ud_f:.3f}, {Uu_f:.3f}]")
        #     print(f"  Min Layer Width: {Lu_f - Ld_f:.4f}, Max Layer Width: {Uu_f - Ud_f:.4f}")
        #     target_w = math.sqrt(t_f - s_f) if (t_f - s_f) > 0 else 0
        #     print(f"  Target Width <= {target_w:.4f}")


    else:
        print("Skipping simulation due to inconsistent initial layer or endpoint parameters.")

    print("\nNote: `sample_from_f1_proposal_inverse_cdf` is still largely a placeholder.")
    print("The quality and efficiency of samples depend heavily on its correct implementation.")

