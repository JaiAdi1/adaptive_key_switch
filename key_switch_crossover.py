import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# ========================================================================
# --- HARDWARE & ARCHITECTURE PARAMETERS ---
# ========================================================================
NUM_ALU = 512
PERM_TPUT = 512
BW_MAX = 460e9       # HBM2 peak bandwidth (460 GB/s)
FREQ = 300e6         # 300 MHz target frequency
N = 2**16
SRAM_CAP = 43e6      # 43 MB exact on-chip capacity

# Total cryptographic bit size required for security
MOD_Q_BITS = 1326
MOD_P_BITS = 306

def format_k(x, pos):
    return f'{int(x/1000)}K'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_k))

    plt.xlabel("Ciphertext Multiplicative Level ($\\ell$)")
    plt.ylabel("Latency (Cycles)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
# ========================================================================
# --- BASE PRIMITIVES (Memory & Fusion Aware) ---
# ========================================================================
def memory_cycles(bytes_rw):
    return (bytes_rw / BW_MAX) * FREQ

def bottleneck(compute, bytes_rw):
    # Hardware hides compute behind memory, or memory behind compute
    return max(compute, memory_cycles(bytes_rw))

def get_ntt_latency(num_limbs, bytes_per_coefficient, parallelism, in_place):
    t_compute = num_limbs * math.log2(N) * (N / (2 * NUM_ALU * parallelism))
    working_set = num_limbs * N * bytes_per_coefficient
    bytes_rw = 0
    if working_set > SRAM_CAP:
        bytes_rw = working_set if in_place else 2 * working_set
    return bottleneck(t_compute, bytes_rw)

def get_bconv_latency(l_in, l_out, num_digits, bytes_per_coefficient, parallelism, fused=False):
    t_compute = (N / (NUM_ALU * parallelism)) * l_in * l_out * num_digits
    t_compute += (N / (NUM_ALU * parallelism)) * l_in * num_digits
    input_bytes = l_in * num_digits * N * bytes_per_coefficient
    output_bytes = l_out * num_digits * N * bytes_per_coefficient
    working_set = input_bytes + output_bytes
    bytes_rw = 0
    if working_set > SRAM_CAP:
        bytes_rw += input_bytes
        if not fused:
            bytes_rw += output_bytes
    return bottleneck(t_compute, bytes_rw)

def get_mod_op_latency(num_ops, bytes_per_coefficient, parallelism, evk_bytes=0):
    t_compute = num_ops / (NUM_ALU * parallelism)
    working_set = num_ops * bytes_per_coefficient
    bytes_rw = evk_bytes
    if working_set > SRAM_CAP:
        bytes_rw += working_set
    return bottleneck(t_compute, bytes_rw)

# ========================================================================
# --- DYNAMIC KEY-SWITCHING LATENCY MODELS ---
# ========================================================================
def calculate_hks(logical_l, dnum, coeff_bits, p):
    b_coeff = coeff_bits / 8
    cap_l = math.ceil(MOD_Q_BITS / coeff_bits) - 1
    k = math.ceil(MOD_P_BITS / coeff_bits)
    low_l = max(0, math.ceil((logical_l / 38) * MOD_Q_BITS / coeff_bits) - 1)

    alpha = math.ceil((cap_l + 1) / dnum)
    beta = math.ceil((low_l + 1) / alpha)

    # iNTTs
    intt_modup = beta * get_ntt_latency(alpha, b_coeff, p, in_place=True)
    intt_moddown = 2 * get_ntt_latency(k, b_coeff, p, in_place=True)

    # NTTs
    ntt_modup =  beta * get_ntt_latency(alpha*(beta-1)+k, b_coeff, p, in_place=True)
    ntt_moddown = 2 * get_ntt_latency(low_l+1, b_coeff, p, in_place=False)

    # BConv (Fused)
    bconv_modup = get_bconv_latency(alpha, alpha*(beta-1)+k, beta, b_coeff, p, fused=True)
    bconv_moddown = get_bconv_latency(k, low_l+1, 2, b_coeff, p, fused=True)

    # Mod Ops
    evk_traffic = beta * 2 * (low_l + 1 + k) * N * b_coeff
    mod_mult = get_mod_op_latency((2)*(beta)*(low_l+1+k)*N, b_coeff, p, evk_bytes=evk_traffic)
    mod_add = get_mod_op_latency(2*(beta-1)*(low_l+1+k)*N, b_coeff, p)
    mod_sub = get_mod_op_latency(2*(low_l+1)*N, b_coeff, p)

    return intt_modup + intt_moddown + ntt_modup + ntt_moddown + bconv_modup + bconv_moddown + mod_mult + mod_add + mod_sub

def calculate_klss(logical_l, gamma, coeff_bits_large, coeff_bits_small, p, v, alpha_tilde, dnum):
    b_coeff_large = coeff_bits_large / 8
    b_coeff_small = coeff_bits_small / 8

    cap_l = math.ceil(MOD_Q_BITS / coeff_bits_small) - 1
    low_l = max(0, math.ceil((logical_l / 38) * MOD_Q_BITS / coeff_bits_small) - 1)

    # alpha = math.ceil((cap_l+1)/gamma)
    alpha = math.ceil((cap_l+1)/dnum)
    beta = math.ceil((low_l + 1) / alpha)

    l_tilde = low_l + 1 + alpha
    # beta_tilde = math.ceil(l_tilde / alpha_tilde)
    beta_tilde = math.ceil(l_tilde / gamma)
    beta_tilde_static = math.ceil((cap_l + 1 + alpha) / gamma)

    #changed the multiplier from 36 to 34 (smallest base multiplier precision use here)
    cap_b = 34 * alpha
    cap_b_tilde = 34 * alpha_tilde
    # alpha_prime = math.ceil(math.log2(2*beta*N*cap_b*cap_b_tilde)/v)
    # alpha_prime = math.ceil((math.log2(2*beta*N)+cap_b+cap_b_tilde)/v)
    alpha_prime = 2

    intt_modup = beta * get_ntt_latency(alpha, b_coeff_small, p, in_place=True)
    intt_moddown = 2 * beta_tilde * get_ntt_latency(alpha_prime, b_coeff_large, p, in_place=True)

    ntt_modup =  beta * get_ntt_latency(alpha_prime, b_coeff_large, p, in_place=True)
    ntt_moddown = 2 * get_ntt_latency(low_l+1, b_coeff_small, p, in_place=False)

    bconv_modup = get_bconv_latency(alpha, alpha_prime, beta, b_coeff_large, p, fused=True)
    chunk_size = math.ceil((low_l + 1 + alpha) / beta_tilde)
    bconv_moddown_1 = get_bconv_latency(alpha_prime, chunk_size, 2 * beta_tilde, b_coeff_large, p, fused=True)

    # l_in to 'alpha' so the matrix math reflects ONLY the limbs being dropped.
    bconv_moddown_2 = get_bconv_latency(alpha, low_l+1, 2, b_coeff_large, p, fused=True)

    evk_traffic = beta * beta_tilde_static * 2 * alpha_prime * N * b_coeff_large
    mod_mult = get_mod_op_latency(2 * alpha_prime * beta * beta_tilde * N, b_coeff_large, p, evk_bytes=evk_traffic)
    mod_add = get_mod_op_latency(2 * alpha_prime * (beta - 1) * beta_tilde * N, b_coeff_large, p)

    # mod_sub to reflect the final ModDown vector subtraction
    mod_sub = get_mod_op_latency(2 * (low_l + 1) * N, b_coeff_small, p)

    return intt_modup + intt_moddown + ntt_modup + ntt_moddown + bconv_modup + bconv_moddown_1 + bconv_moddown_2 + mod_mult + mod_add + mod_sub

# ========================================================================
# --- EVALUATION SCRIPT: CLEAN BASELINE WITH CROSSOVER ---
# ========================================================================
if __name__ == "__main__":
    logical_levels = 38

    # Architecture's native multi-precision parameters
    target_dnum = 3        # HKS optimal chunking
    target_gamma = 9       # KLSS optimal chunking
    alpha_tilde_val = 6
    v_val = 60

    l_values = list(range(1, logical_levels + 1))

    print("\n" + "="*80)
    print(" SECTION IV: FUSION-AWARE BASELINE PARAMETERS")
    print("="*80)
    print(f"Modulus Q (Bits):             {MOD_Q_BITS}")
    print(f"Modulus P (Bits):             {MOD_P_BITS}")
    print(f"HKS dnum:                     {target_dnum}")
    print(f"KLSS gamma:                   {target_gamma}")
    print(f"KLSS alpha_tilde:             {alpha_tilde_val}")
    print("="*80)

    # Calculate natively area-optimized HKS (34-bit, p=2.0)
    h_costs = [calculate_hks(l, target_dnum, coeff_bits=34, p=2.0) for l in l_values]

    # Calculate natively precision-optimized KLSS (64-bit, p=1.0)
    k_costs = [calculate_klss(l, target_gamma, coeff_bits_large=62, coeff_bits_small=34, p=1.0, v=62, alpha_tilde=alpha_tilde_val, dnum=target_dnum) for l in l_values]

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 4))

    # Plot dots instead of connected lines
    plt.plot(l_values, h_costs, marker='o', linestyle='', markersize=6,
            label=f"HKS Latency",
            color='#0284c7')

    plt.plot(l_values, k_costs, marker='o', linestyle='', markersize=6,
            label=f"KLSS Latency",
            color='#ea580c')

    # Regressions (trend lines)
    hks_poly = np.polyfit(l_values, h_costs, 2)
    plt.plot(l_values, np.polyval(hks_poly, l_values),
            color='#0369a1', linestyle='-', linewidth=1.5, label='HKS Regression')

    klss_poly = np.polyfit(l_values, k_costs, 1)
    plt.plot(l_values, np.polyval(klss_poly, l_values),
            color='#9a3412', linestyle='-', linewidth=1.5, label='KLSS Regression')

    # Axis labels
    plt.xlabel("Ciphertext Multiplicative Level ($\\ell$)", fontsize=12, fontweight='bold')
    plt.ylabel("Latency (cycle)", fontsize=12, fontweight='bold')

    plt.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
    plt.minorticks_on()

    # Format Y-axis as 100K, 200K...
    def format_k(x, pos):
        return f'{int(x/1000)}K'

    plt.gca().yaxis.set_major_formatter(FuncFormatter(format_k))

    # Optional: draw switching point line (edit x value if needed)
    # switch_point = 28
    # plt.axvline(x=switch_point, linestyle='--', color='gray', linewidth=1)
    # plt.text(switch_point + 0.5, min(h_costs), 'Switch Point', rotation=90, color='gray')

    # Style
    plt.legend(fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("./keyswitch_tradeoff.pdf", bbox_inches='tight')
    # plt.show()
