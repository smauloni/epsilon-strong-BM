import math
import matplotlib.pyplot as plt
from typing import Tuple
import random
random.seed(42)  # For reproducibility
import scipy.optimize
MAX_REFINE_ITER = 10


# === helper functions for alternating sums and zeta functions ==========
def sigma_bar(j: int, x: float, y: float, delta: float, xi: float, ell: float) -> float:
    """
    Eq. (4.2) first term:
    exp(-2/ell * (delta*j + xi - x) * (delta*j + xi - y))
    """
    return math.exp(-2.0 / ell * (delta * j + xi - x) * (delta * j + xi - y))

def tau_bar(j: int, x: float, y: float, delta: float, ell: float) -> float:
    """
    Eq. (4.2) second term:
    exp(-2.0*j/ell * (delta**2 * j + delta*(x - y)))
    """
    return math.exp(-2.0 * j / ell * (delta * delta * j + delta * (x - y)))

def alternating_pair(L: float, U: float, ell: float,
                     x: float, y: float, K: int) -> Tuple[float, float]:
    """
    Upper and lower alternating sums S_{2K-1} and S_{2K} truncated at j=1..K.
    """
    delta = U - L
    s = 0.0
    S_upper = 0.0
    S_lower = 0.0

    for j in range(1, K + 1):
        sigma_j = (
            sigma_bar(j, x, y, delta, L, ell)
            + sigma_bar(j, -x, -y, delta, -U, ell)
        )
        tau_j = (
            tau_bar(j, x, y, delta, ell)
            + tau_bar(j, -x, -y, delta, ell)
        )

        S_upper = s + sigma_j
        S_lower = S_upper - tau_j
        s += sigma_j - tau_j

    # clamp to [0,1]
    S_upper = max(0.0, min(1.0, S_upper))
    S_lower = max(0.0, min(1.0, S_lower))
    return S_upper, S_lower

def zeta(L: float, U: float, ell: float, x: float, y: float,
         max_iter: int = 1000, tol: float = 1e-12) -> float:
    """
    Infinite sum ζ = Σ_{j=1}∞ [sigma_j - tau_j], stopping when terms < tol.
    """
    if not (L < x < U and L < y < U):
        return 1.0

    delta = U - L
    total = 0.0

    for j in range(1, max_iter + 1):
        sigma_j = (
            sigma_bar(j, x, y, delta, L, ell)
            + sigma_bar(j, -x, -y, delta, -U, ell)
        )
        tau_j = (
            tau_bar(j, x, y, delta, ell)
            + tau_bar(j, -x, -y, delta, ell)
        )

        term = sigma_j - tau_j
        total += term
        if abs(term) < tol * max(1.0, abs(total)):
            break

    return max(0.0, min(1.0, total))

def gamma_func(L: float, U: float, ell: float, x: float, y: float) -> float:
    """
    Y = 1 - ζ, with boundary case L >= U.
    """
    if L >= U:
        # if degenerate and x=y=L=U, stays in; else never in
        return 1.0 if (x == y == L == U) else 0.0
    return 1.0 - zeta(L, U, ell, x, y)

def beta_func(L_low: float, L_high: float, U_low: float, U_high: float,
              ell: float, x: float, y: float) -> float:
    """
    Eq. (4.8): β = gamma(L_low,U_high) - gamma(L_high,U_high)
               - gamma(L_low,U_low) + gamma(L_high,U_low)
    """
    if not (L_low < L_high < U_low < U_high):
        return 0.0

    g = gamma_func
    return (
        g(L_low,  U_high, ell, x, y)
        - g(L_high, U_high, ell, x, y)
        - g(L_low,  U_low,  ell, x, y)
        + g(L_high, U_low,  ell, x, y)
    )

def rho_func(L_down: float, L_up: float, U_down: float, U_up: float,
             q: float, r: float, x: float, w: float, y: float) -> float:
    """
    Eq. (4.9): rho = g1*g2 - g3*g4 - g5*g6 + g7*g8,
    where each gi = gamma(·,·;·,·,·) on the two sub-bridges.
    """
    if not (L_down < L_up <= U_down < U_up):
        return 0.0

    g = gamma_func
    return (
        g(L_down, U_up,   q, x, w) * g(L_down, U_up,   r, w, y)
        - g(L_up,   U_up,   q, x, w) * g(L_up,   U_up,   r, w, y)
        - g(L_down, U_down, q, x, w) * g(L_down, U_down, r, w, y)
        + g(L_up,   U_down, q, x, w) * g(L_up,   U_down, r, w, y)
    )


# === Gaussian Density for rejection sampler ============================
def pi_unnormalized(w: float, s: float, t: float, Xs: float, Xt: float) -> float:
    """Eq.(5.1) Gaussian density (up to const) of midpoint w at time (s+t)/2."""
    if t <= s:
        # degenerate: only mass at w==Xs==Xt when s==t
        return 1.0 if (t == s and w == Xs == Xt) else 0.0

    q = r = (t - s)/2
    # conditional mean & variance
    mu = (r*Xs + q*Xt)/(q + r)
    var = q*r/(q + r)
    if var <= 0:
        return 1.0 if w == mu else 0.0

    # unnormalized Gaussian
    exponent = -0.5 * (w - mu)**2 / var
    return math.exp(exponent)

def SZ_rho_bound(params: tuple, n: int, upper: bool) -> float:
    """
    Finite-sum bound on rho (Section 4), with n+1 terms.
    params = (Ld, Lu, Ud, Uu, q, r, x, w, y)
    """
    Ld, Lu, Ud, Uu, q, r, x, w, y = params
    if not (Ld < Lu <= Ud < Uu):
        return 0.0

    # the 8 intervals we need zeta‐bounds on
    zones = [
        (Ld, Uu, q, x, w), (Ld, Uu, r, w, y),
        (Lu, Uu, q, x, w), (Lu, Uu, r, w, y),
        (Ld, Ud, q, x, w), (Ld, Ud, r, w, y),
        (Lu, Ud, q, x, w), (Lu, Ud, r, w, y),
    ]
    K = n + 1
    # directly call alternating_pair from this file
    S_up, S_lo = zip(*(alternating_pair(L, U, ell, x0, y0, K)
                       for (L, U, ell, x0, y0) in zones))

    def block(i, j):
        if upper:
            return 1.0 - S_lo[i] - S_lo[j] + S_up[i] * S_up[j]
        else:
            return 1.0 - S_up[i] - S_up[j] + S_lo[i] * S_lo[j]

    A = block(0, 1)
    B = block(2, 3)
    C = block(4, 5)
    D = block(6, 7)
    val = A - B - C + D
    return max(0.0, min(1.0, val))

def f1_integrand(u: float, Ld: float, Lu: float, Ud: float, Uu: float,
                 q: float, r: float, Xs: float, Xt: float,
                 s: float, t: float, n_SZ: int) -> float:
    pi_val = pi_unnormalized(u, s, t, Xs, Xt)
    if pi_val == 0.0:
        return 0.0
    rho_args = (Ld, Lu, Ud, Uu, q, r, Xs, u, Xt)
    return pi_val * SZ_rho_bound(rho_args, n_SZ, upper=True)

def F1_cdf_and_norm(w: float,
                    Ld: float, Lu: float, Ud: float, Uu: float,
                    q: float, r: float, Xs: float, Xt: float) -> Tuple[float, float]:
    """
    Returns (F1(w), C) where
      C = ∫_{Ld}^{Uu} f1(u) du
      F1(w) = (∫_{Ld}^{min(w,Uu)} f1(u) du) / C
    all in closed form via Eq. (5.5).
    """
    # sanitize trivial
    if Uu <= Ld:
        C = 1.0
        return (0.0 if w < Ld else 1.0), C

    mu     = (r*Xs + q*Xt)/(q+r)
    sigma2 = q*r/(q+r)
    sigma  = math.sqrt(sigma2)
    Phi    = lambda z: 0.5*(1 + math.erf(z/math.sqrt(2)))

    # coefficients: list of tuples (sign, Li, Ui, bi, ai)
    specs = [
      (+1, Ld, Uu,
          -2*((Xs-Ld)/q + (Xt-Ld)/r),
           2*((Xs-Ld)/q + (Xt-Ld)/r)*Ld),
      (-1, Lu, Uu,
          -2*((Xs-Lu)/q + (Xt-Lu)/r),
           2*((Xs-Lu)/q + (Xt-Lu)/r)*Lu),
      (-1, Ld, Ud,
          -2*((Xs-Ld)/q + (Xt-Ld)/r),
           2*((Xs-Ld)/q + (Xt-Ld)/r)*Ld),
      (+1, Lu, Ud,
          -2*((Xs-Lu)/q + (Xt-Ld)/r),
           2*((Xs-Lu)/q * Lu + (Xt-Ld)/r * Ld)),
    ]

    def segment(sign, Li, Ui, bi, ai, upper):
        """Compute sign * exp(ai + bi*mu + .5*bi²*sigma2) * [Φ(z(upper))−Φ(z(Li))]"""
        if upper <= Li:
            return 0.0
        hi = min(upper, Ui)
        z  = lambda x: (x - (mu + bi*sigma2)) / sigma
        pref = math.exp(ai + bi*mu + 0.5*bi*bi*sigma2)
        return sign * pref * (Phi(z(hi)) - Phi(z(Li)))

    # numerator: ∫ f1 from Ld to w
    num = sum(segment(sign, Li, Ui, bi, ai, w) for sign,Li,Ui,bi,ai in specs)
    # denominator: ∫ f1 from Ld to Uu
    C   = sum(segment(sign, Li, Ui, bi, ai, Ui) for sign,Li,Ui,bi,ai in specs)

    F1 = num / C if C > 0 else 0.0
    return F1, C

def sample_f1_inverse(s: float, t: float, x_s: float, x_t: float,
                       Ld: float, Lu: float, Ud: float, Uu: float,
                       w_min: float, w_max: float, n_zeta_terms: int) -> float:
    """
    Draw midpoint w from f1 by inversion: F1(w) = R/C,
    where C is obtained in closed form.
    """
    # compute durations
    dt = t - s
    if dt <= 0:
        return x_s

    q = r = dt/2

    # 1) get normalization C via closed form
    _, C = F1_cdf_and_norm(Uu, Ld, Lu, Ud, Uu, q, r, x_s, x_t)
    if C <= 0 or not math.isfinite(C):
        # fallback uniform
        return random.uniform(Ld, Uu)

    # 2) draw a uniform mass in [0, C]
    target_mass = random.random() * C

    # 3) invert F1(w) * C = target_mass  ⇒  F1(w) = target_mass / C
    target_cdf = target_mass / C
    try:
        w_star = scipy.optimize.brentq(
            lambda w: F1_cdf_and_norm(w, Ld, Lu, Ud, Uu, q, r, x_s, x_t)[0] - target_cdf,
            Ld, Uu, xtol=1e-9, rtol=1e-9, maxiter=100
        )
        return w_star
    except ValueError:
        # bracket failure → uniform fallback
        print("Warning: failed, falling back to uniform sampling.")
        return random.uniform(Ld, Uu)

def compare_rho(K: float, params: tuple, max_iter: int = 50) -> bool:
    for n in range(max_iter):
        lo = SZ_rho_bound(params, n, upper=False)
        hi = SZ_rho_bound(params, n, upper=True)
        if   K < lo: return True
        elif K > hi: return False
        if hi - lo < 1e-12:
            return (K < (lo + hi) / 2)
    return K < SZ_rho_bound(params, max_iter - 1, upper=False)


# ── Helpers ────────────────────────────────────────────────────────────
def clamp_unit(x: float) -> float:
    return max(0.0, min(1.0, x))

def sanitize_interval(low: float, high: float, eps: float = 1e-9) -> Tuple[float, float]:
    """Ensure low < high by at least eps; if not, expand around mid."""
    if low >= high - eps:
        mid = (low + high) / 2.0
        low, high = mid - eps, mid + eps
    return low, high

def update_bounds_with_point(bounds: Tuple[float, float], point: float) -> Tuple[float, float]:
    """Given (low, high) or (None,None), refine with point."""
    low, high = bounds
    if low is None or point > low:
        low = point if low is None else low
    if high is None or point < high:
        high = point if high is None else high
    return low, high
# ───────────────────────────────────────────────────────────────────────


# === Sampling Midpoint, Update and Bisect Layers (Section 5.1-5.2) =====
def sample_midpoint(s: float, t: float, x_s: float, x_t: float,
                    Ld: float=None, Lu: float=None,
                    Ud: float=None, Uu: float=None,
                    max_reject: int=1000, n_zeta_terms: int=0) -> Tuple[float, float, float, float, float]:
    """
    Return (w_mid, Ld', Lu', Ud', Uu').
    """
    dt = t - s
    if dt <= 0:
        # degenerate: mid = start
        w_mid = x_s
        return w_mid, x_s, x_s, x_s, x_s

    # initial valid domain for w
    default_mu = 0.5*(x_s + x_t)
    default_sigma = math.sqrt(dt/4)
    w_min = Ld if Ld is not None else default_mu - 5*default_sigma
    w_max = Uu if Uu is not None else default_mu + 5*default_sigma
    w_min, w_max = sanitize_interval(w_min, w_max)

    # precompute durations
    q = r = dt/2

    for _ in range(max_reject):
        # 1) draw candidate via inverse‐CDF proposal
        w = sample_f1_inverse(s, t, x_s, x_t, Ld, Lu, Ud, Uu, w_min, w_max, n_zeta_terms)
        pi_val = pi_unnormalized(w, s, t, x_s, x_t)
        if pi_val <= 0:
            continue

        # 2) adjust layers around w
        Ld_new, Lu_new = update_bounds_with_point((Ld, Lu), w)
        Ud_new, Uu_new = update_bounds_with_point((Ud, Uu), w)
        Ld_new, Lu_new = sanitize_interval(Ld_new, Lu_new)
        Ud_new, Uu_new = sanitize_interval(Ud_new, Uu_new)

        # 3) compute accept‐prob bound SZ_rho
        rho_params = (Ld_new, Lu_new, Ud_new, Uu_new, q, r, x_s, w, x_t)
        sz_bound = SZ_rho_bound(rho_params, n_zeta_terms, upper=True)
        if sz_bound <= 0:
            continue

        # 4) accept/reject
        if compare_rho(random.random() * sz_bound, rho_params):
            return w, Ld_new, Lu_new, Ud_new, Uu_new

    # fallback to midpoint
    w_mid = 0.5*(x_s + x_t)
    return w_mid, *sanitize_interval(w_mid, w_mid), *sanitize_interval(w_mid, w_mid)

def update_layers(s: float, t: float,
                  x_s: float, x_mid: float, x_t: float,
                  Ld: float, Lu: float, Ud: float, Uu: float):
    """
    §5.2: Given a midpoint x_mid on [s,t], pick one of 3x3 scenarios
    for how the min-interval and max.interval split across left/right.
    """

    # Degenerate or invalid global layer: just split by values
    if not (t > s and Ld < Lu and Ud < Uu and Lu <= Ud):
        return ((min(x_s, x_mid), max(x_s, x_mid)),
                (min(x_mid, x_t), max(x_mid, x_t)))

    # Precompute durations
    t_mid = 0.5*(s + t)
    q = t_mid - s
    r = t - t_mid

    # All 3 choices for left‐min interval, and 3 for right‐max interval
    left_choices  = [(Ld, Lu), (Ld, x_mid), (x_s, x_mid)]
    right_choices = [(Ud, Uu), (x_mid, Uu), (Ud, x_t)]

    scenarios = []
    weights   = []

    for (LdL, LuL) in left_choices:
        # ensure sanitized
        LdL, LuL = sanitize_interval(LdL, LuL)
        for (UdR, UuR) in right_choices:
            UdR, UuR = sanitize_interval(UdR, UuR)

            # compute P(min in [LdL,LuL]) * P(max in [UdR,UuR])
            p_min = beta_func(LdL, LuL, Ud, Uu, q, x_s, x_mid) if (LdL < LuL) else 0.0
            p_max = beta_func(Ld,  Lu,  UdR, UuR,  r, x_mid, x_t) if (UdR < UuR) else 0.0

            scenarios.append((LdL, LuL, UdR, UuR))
            weights.append(p_min * p_max)

    # normalize weights (avoid all-zero)
    total = sum(weights)
    if total <= 0:
        chosen = 0
    else:
        # use random.choices for a weighted pick
        norm_weights = [w/total for w in weights]
        chosen = random.choices(range(len(scenarios)), weights=norm_weights, k=1)[0]

    Ld_s, Lu_s, Ud_r, Uu_r = scenarios[chosen]
    # final sanitize to guarantee strict ordering
    Ld_s, Lu_s = sanitize_interval(Ld_s, Lu_s)
    Ud_r, Uu_r = sanitize_interval(Ud_r, Uu_r)

    return (Ld_s, Lu_s), (Ud_r, Uu_r)

def bisect_layer(s: float, t: float, x_s: float, x_t: float,
                 Ld: float, Lu: float, Ud: float, Uu: float):
    """
    §5.1-5.2: Sample midpoint + update layers, returning two full 8-tuples.
    """
    if t <= s:
        return None, None

    # 1) sample midpoint + updated global layers
    w_mid, Ld2, Lu2, Ud2, Uu2 = sample_midpoint(s, t, x_s, x_t, Ld, Lu, Ud, Uu)

    # 2) refine into left and right intervals
    t_mid = 0.5 * (s + t)
    (Ld_s, Lu_s), (Ud_r, Uu_r) = update_layers(s, t, x_s, w_mid, x_t,
                                               Ld2, Lu2, Ud2, Uu2)

    # 3) pack into full layer info
    left_info = (s,       t_mid, x_s,  w_mid,
                 Ld_s,    Lu_s, Ud_r,  Uu_r)
    right_info= (t_mid,   t,     w_mid, x_t,
                 Ld_s,    Lu_s, Ud_r,  Uu_r)

    return left_info, right_info


# === Refinement (Section 5.3) ==========================================
def SZ_n_beta_bound(beta_params: tuple,
                    n: int,
                    get_upper_beta_bound: bool) -> float:
    """
    Alternating-series bound SZ_n for the beta function.
    
    beta_params = (L_low, L_high, U_low, U_high,
                   interval_length, x_s, x_t)
    If get_upper_beta_bound is True, returns the (2n+1)-term upper bound;
    otherwise returns the 2n-term lower bound.
    """
    L_low, L_high, U_low, U_high, length, x_s, x_t = beta_params

    # invalid ordering ⇒ beta = 0
    if not (L_low < L_high and U_low < U_high and L_high <= U_low):
        return 0.0

    # we truncate after K = n+1 zeta‐terms
    K = n + 1

    # helper to get zeta bounds on [a,b] with ends (x,y)
    # uses your alternating_pair(L, U, ell, x, y, K) → (S_up, S_lo)
    def zeta_bounds(a, b, x, y):
        return alternating_pair(a, b, length, x, y, K)

    # gather (S_up, S_lo) for the four zeta‐calls
    # A: zeta(L_low,  U_high)
    # B: zeta(L_high, U_high)
    # C: zeta(L_low,  U_low)
    # D: zeta(L_high, U_low)
    S_u_A, S_l_A = zeta_bounds(L_low,  U_high, x_s, x_t)
    S_u_B, S_l_B = zeta_bounds(L_high, U_high, x_s, x_t)
    S_u_C, S_l_C = zeta_bounds(L_low,  U_low,  x_s, x_t)
    S_u_D, S_l_D = zeta_bounds(L_high, U_low,  x_s, x_t)

    if get_upper_beta_bound:
        # upper bound =  zeta_B + zeta_C  - zeta_A - zeta_D
        # use upper for positive terms, lower for negative
        bound = (S_u_B + S_u_C) - (S_l_A + S_l_D)
    else:
        # lower bound =  zeta_B + zeta_C  - zeta_A - zeta_D
        # use lower for positive terms, upper for negative
        bound = (S_l_B + S_l_C) - (S_u_A + S_u_D)

    # clamp into [0,1]
    return max(0.0, min(1.0, bound))

def compare_beta(K: float,
                 beta_params: tuple,
                 max_iter: int = 50,
                 tol: float = 1e-12) -> bool:
    """
    Return True if K < β(beta_params), False otherwise,
    by iteratively computing SZ_n_beta_bound lower/upper bounds.
    
    beta_params = (min_low, min_high, max_low, max_high,
                   interval_length, x_s, x_t)
    """
    Ld, Lu, Ud, Uu, length, x_s, x_t = beta_params
    # invalid layer ⇒ β = 0 ⇒ K < β is always False
    if not (Ld < Lu and Ud < Uu and Lu <= Ud):
        return False

    for n in range(max_iter):
        lo = SZ_n_beta_bound(beta_params, n, get_upper_beta_bound=False)
        hi = SZ_n_beta_bound(beta_params, n, get_upper_beta_bound=True)
        if   K < lo - tol: return True
        elif K > hi + tol: return False
        if hi - lo < tol:
            return K < (lo + hi)/2
    # final fallback
    final_lo = SZ_n_beta_bound(beta_params, max_iter-1, get_upper_beta_bound=False)
    return K < final_lo - tol

def refine_layer(s: float, t: float, x_s: float, x_t: float,
                 Ld: float, Lu: float, Ud: float, Uu: float,
                 mode: str="min"):
    length = t - s
    if length <= 0:
        return Ld, Lu, Ud, Uu

    # choose midpoint of the interval to refine
    if mode == "min":
        mid = 0.5*(Ld + Lu)
        K = random.random() * beta_func(Ld, Lu, Ud, Uu, length, x_s, x_t)
        if compare_beta(K, (mid, Lu, Ud, Uu, length, x_s, x_t)):
            Ld = mid
        else:
            Lu = mid

    elif mode == "max":
        mid = 0.5*(Ud + Uu)
        K = random.random() * beta_func(Ld, Lu, Ud, Uu, length, x_s, x_t)
        if compare_beta(K, (Ld, Lu, mid, Uu, length, x_s, x_t)):
            Ud = mid
        else:
            Uu = mid

    else:
        raise ValueError("mode must be 'min' or 'max'")

    # sanitize
    Ld, Lu = sanitize_interval(Ld, Lu)
    Ud, Uu = sanitize_interval(Ud, Uu)
    return Ld, Lu, Ud, Uu


# === Epsilon-Strong Algorithm (Table 2) ================================
def epsilon_strong(x0: float,
                   x1: float,
                   min_low: float, min_high: float,
                   max_low: float, max_high: float,
                   num_bisects: int) -> list:
    """
    Returns the list of 2^num_bisects final intersection layers
    Each layer is (s, t, x_s, x_t, min_low, min_high, max_low, max_high).
    """
    # initialize
    layers = [(0.0, 1.0, x0, x1,
               *sanitize_interval(min_low, min_high),
               *sanitize_interval(max_low, max_high))]

    for iteration in range(1, num_bisects + 1):
        next_layers = []
        for (s, t, xs, xt, ml, mh, ML, MH) in layers:
            # skip degenerate
            if t - s < 1e-9:
                next_layers.append((s, t, xs, xt, ml, mh, ML, MH))
                continue

            # 1) bisect & sample midpoint + update layers
            left_info, right_info = bisect_layer(s, t, xs, xt, ml, mh, ML, MH)
            for (u, v, xu, xv, ml2, mh2, ML2, MH2) in (left_info, right_info):
                # 2) refine min until width <= sqrt(interval)
                target = math.sqrt(v - u)
                for _ in range(MAX_REFINE_ITER):
                    if mh2 - ml2 <= target + 1e-9:
                        break
                    ml2, mh2, ML2, MH2 = refine_layer(u, v, xu, xv, ml2, mh2, ML2, MH2, mode="min")
                # 3) refine max similarly
                for _ in range(MAX_REFINE_ITER):
                    if MH2 - ML2 <= target + 1e-9:
                        break
                    ml2, mh2, ML2, MH2 = refine_layer(u, v, xu, xv, ml2, mh2, ML2, MH2, mode="max")

                next_layers.append((u, v, xu, xv,
                                    *sanitize_interval(ml2, mh2),
                                    *sanitize_interval(ML2, MH2)))
        layers = next_layers
        print(f"Iteration {iteration}/{num_bisects}: {len(layers)} layers")

    return layers


# === Plotting Function =================================================
def plot_layers(layers: list, x0: float, x1: float, title: str = "Epsilon-Strong Layers"):
    """
    Each layer: (s, t, xs, xt, ml, mh, ML, MH).
    Plots min_low (ml) in blue and max_high (MH) in red, plus the sampled path.
    """
    if not layers:
        print("No layers to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    # Draw bounds
    for (s, t, xs, xt, ml, mh, ML, MH) in sorted(layers, key=lambda L: L[0]):
        ax.hlines(ml, s, t, color="blue", linewidth=1.5, alpha=0.6)
        ax.hlines(MH, s, t, color="red",  linewidth=1.5, alpha=0.6)

    # Extract and plot the piecewise path
    times, values = [], []
    sorted_layers = sorted(layers, key=lambda L: L[0])
    times.append(sorted_layers[0][0]); values.append(sorted_layers[0][2])
    for (s, t, xs, xt, *_rest) in sorted_layers:
        times.append(t); values.append(xt)
    ax.plot(times, values, "ko-", markersize=4, label="Bridge Sample")

    ax.set_xlabel("Time"); ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(True, linestyle=":")
    ax.legend()
    plt.tight_layout()
    plt.show()




# === Example Usage ===
if __name__ == "__main__":
    # endpoints and number of bisections to perform
    x0, x1 = 0.0, 0.5
    num_bisects = 5
    # initial bounds on min and max
    min_low, min_high = -0.5, 0.0
    max_low, max_high =  0.05, 0.6

    # run n bisections → 2^n layers
    final_layers = epsilon_strong(x0, x1, min_low, min_high, max_low, max_high, num_bisects=num_bisects)
    plot_layers(final_layers, x0, x1, title=f"Epsilon-Strong (n={num_bisects} Layers)")
