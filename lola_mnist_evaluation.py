import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ========================================================================
# --- HARDWARE & ARCHITECTURE PARAMETERS (UPDATED) ---
# ========================================================================
NUM_ALU = 512
PERM_TPUT = 512
BW_MAX = 460e9
FREQ = 300e6
N = 2**14
SRAM_CAP = 43e6

# MOD_Q_BITS = 544
# MOD_P_BITS = 136
MOD_Q_BITS = 210
MOD_P_BITS = 34
L_MAX = math.ceil(MOD_Q_BITS/34)

# ========================================================================
# --- BASE PRIMITIVES (SPU-OPTIMIZED) ---
# ========================================================================
def memory_cycles(bytes_rw):
    return (bytes_rw / BW_MAX) * FREQ

def bottleneck(compute, bytes_rw):
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
# --- SPLIT MODELS FOR HOISTING OPTIMIZATION ---
# ========================================================================
def hks_modup_cost(logical_l, dnum, coeff_bits, p):
    b_coeff = coeff_bits / 8
    cap_l = math.ceil(MOD_Q_BITS / coeff_bits) - 1
    k = math.ceil(MOD_P_BITS / coeff_bits)
    low_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / coeff_bits) - 1)

    alpha = math.ceil((cap_l + 1) / dnum)
    beta = math.ceil((low_l + 1) / alpha)

    intt_modup = beta * get_ntt_latency(alpha, b_coeff, p, in_place=True)
    ntt_modup =  beta * get_ntt_latency(alpha*(beta-1)+k, b_coeff, p, in_place=True)
    bconv_modup = get_bconv_latency(alpha, alpha*(beta-1)+k, beta, b_coeff, p, fused=True)

    return intt_modup + ntt_modup + bconv_modup

def hks_mult_down_cost(logical_l, dnum, coeff_bits, p):
    b_coeff = coeff_bits / 8
    cap_l = math.ceil(MOD_Q_BITS / coeff_bits) - 1
    k = math.ceil(MOD_P_BITS / coeff_bits)
    low_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / coeff_bits) - 1)

    alpha = math.ceil((cap_l + 1) / dnum)
    beta = math.ceil((low_l + 1) / alpha)

    evk_traffic = beta * 2 * (low_l + 1 + k) * N * b_coeff
    mod_mult = get_mod_op_latency((2)*(beta)*(low_l+1+k)*N, b_coeff, p, evk_bytes=evk_traffic)
    mod_add = get_mod_op_latency(2*(beta-1)*(low_l+1+k)*N, b_coeff, p)

    intt_moddown = 2 * get_ntt_latency(k, b_coeff, p, in_place=True)
    bconv_moddown = get_bconv_latency(k, low_l+1, 2, b_coeff, p, fused=True)
    mod_sub = get_mod_op_latency(2*(low_l+1)*N, b_coeff, p)
    ntt_moddown = 2 * get_ntt_latency(low_l+1, b_coeff, p, in_place=False)

    return mod_mult + mod_add + intt_moddown + bconv_moddown + mod_sub + ntt_moddown

def klss_modup_cost(logical_l, gamma, coeff_bits_large, coeff_bits_small, p, v, alpha_tilde, dnum):
    b_coeff_large = coeff_bits_large / 8
    b_coeff_small = coeff_bits_small / 8

    cap_l = math.ceil(MOD_Q_BITS / coeff_bits_small) - 1
    low_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / coeff_bits_small) - 1)

    alpha = math.ceil((cap_l+1)/dnum)
    beta = math.ceil((low_l + 1) / alpha)

    alpha_prime = 2

    intt_modup = beta * get_ntt_latency(alpha, b_coeff_small, p, in_place=True)
    ntt_modup =  beta * get_ntt_latency(alpha_prime, b_coeff_large, p, in_place=True)
    bconv_modup = get_bconv_latency(alpha, alpha_prime, beta, b_coeff_large, p, fused=True)

    return intt_modup + ntt_modup + bconv_modup

def klss_mult_down_cost(logical_l, gamma, coeff_bits_large, coeff_bits_small, p, v, alpha_tilde, dnum):
    b_coeff_large = coeff_bits_large / 8
    b_coeff_small = coeff_bits_small / 8

    cap_l = math.ceil(MOD_Q_BITS / coeff_bits_small) - 1
    low_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / coeff_bits_small) - 1)

    alpha = math.ceil((cap_l+1)/dnum)
    beta = math.ceil((low_l + 1) / alpha)
    l_tilde = low_l + 1 + alpha
    beta_tilde = math.ceil(l_tilde / gamma)
    beta_tilde_static = math.ceil((cap_l + 1 + alpha) / gamma)
    alpha_prime = 2

    evk_traffic = beta * beta_tilde_static * 2 * alpha_prime * N * b_coeff_large
    mod_mult = get_mod_op_latency(2 * alpha_prime * beta * beta_tilde * N, b_coeff_large, p, evk_bytes=evk_traffic)
    mod_add = get_mod_op_latency(2 * alpha_prime * (beta - 1) * beta_tilde * N, b_coeff_large, p)

    intt_moddown = 2 * beta_tilde * get_ntt_latency(alpha_prime, b_coeff_large, p, in_place=True)

    chunk_size = math.ceil((low_l + 1 + alpha) / beta_tilde)
    bconv_moddown_1 = get_bconv_latency(alpha_prime, chunk_size, 2 * beta_tilde, b_coeff_large, p, fused=True)
    bconv_moddown_2 = get_bconv_latency(alpha, low_l+1, 2, b_coeff_large, p, fused=True)

    mod_sub = get_mod_op_latency(2 * (low_l + 1) * N, b_coeff_small, p)
    ntt_moddown = 2 * get_ntt_latency(low_l+1, b_coeff_small, p, in_place=False)

    return mod_mult + mod_add + intt_moddown + bconv_moddown_1 + bconv_moddown_2 + mod_sub + ntt_moddown

# ========================================================================
# --- DYNAMIC COMPILER LOGIC ---
# ========================================================================
def get_hoisted_rotations(logical_l, r, mode, target_dnum, target_gamma, v=62, alpha_tilde=6, trace=False):
    h_cost = hks_modup_cost(logical_l, target_dnum, 34, 2.0) + r * hks_mult_down_cost(logical_l, target_dnum, 34, 2.0)
    k_cost = klss_modup_cost(logical_l, target_gamma, 62, 34, 1.0, v, alpha_tilde, target_dnum) + r * klss_mult_down_cost(logical_l, target_gamma, 62, 34, 1.0, v, alpha_tilde, target_dnum)

    low_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)
    base_add = r * get_mod_op_latency(2*(low_l+1)*N, 34/8, 2.0)

    picked = "HKS" if h_cost < k_cost else "KLSS"
    if trace and mode == 'hybrid':
        print(f"    [Level {logical_l:02d} | Rotations (r={r:03d})] HKS: {h_cost:,.0f} | KLSS: {k_cost:,.0f} -> Picked {picked}")

    return (h_cost if mode == 'hks' else k_cost if mode == 'klss' else min(h_cost, k_cost)) + base_add

def get_cmult(logical_l, mode, target_dnum, target_gamma, v=62, alpha_tilde=6, trace=False):
    h_cost = hks_modup_cost(logical_l, target_dnum, 34, 2.0) + hks_mult_down_cost(logical_l, target_dnum, 34, 2.0)
    k_cost = klss_modup_cost(logical_l, target_gamma, 62, 34, 1.0, v, alpha_tilde, target_dnum) + klss_mult_down_cost(logical_l, target_gamma, 62, 34, 1.0, v, alpha_tilde, target_dnum)

    low_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)
    base = 3*get_mod_op_latency(2*(low_l+1)*N, 34/8, 2.0) + get_mod_op_latency(2*(low_l+1)*N, 34/8, 2.0)

    picked = "HKS" if h_cost < k_cost else "KLSS"
    if trace and mode == 'hybrid':
        print(f"    [Level {logical_l:02d} | CMult   (Square) ] HKS: {h_cost:,.0f}     | KLSS: {k_cost:,.0f}     -> Picked {picked}")

    return base + (h_cost if mode == 'hks' else k_cost if mode == 'klss' else min(h_cost, k_cost))

def get_rescale(logical_l):
    low_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)
    return get_mod_op_latency(2*(low_l)*N, 34/8, 2.0)

def get_pmult(logical_l):
    low_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)
    return get_mod_op_latency(2*(low_l+1)*N, 34/8, 2.0)

# ========================================================================
# --- LOLA-MNIST NETWORK SIMULATION ---
# ========================================================================
def sim_lola(mode, target_dnum, target_gamma, trace=False):
    layers = [
        {"name": "Conv1", "rotates": 150, "pmults": 150, "cadds": 150},
        {"name": "Square1", "cmults": 1},
        {"name": "Dense1", "rotates": 200, "pmults": 200, "cadds": 200},
        {"name": "Square2", "cmults": 1},
        {"name": "Dense2", "rotates": 62, "pmults": 62, "cadds": 62},
    ]

    # logical_l = 14 # LoLa operates at shallow depths
    logical_l = L_MAX
    cyc = 0
    for layer in layers:
        if trace and mode == 'hybrid': print(f"  --- {layer['name']} ---")
        if "rotates" in layer:
            r = layer["rotates"]
            cyc += r * get_pmult(logical_l) + get_rescale(logical_l)
            logical_l -= 1
            cyc += get_hoisted_rotations(logical_l, r, mode, target_dnum, target_gamma, trace=trace)
        else:
            cyc += layer["cmults"] * get_cmult(logical_l, mode, target_dnum, target_gamma, trace=trace) + get_rescale(logical_l)
            logical_l -= 1
    return cyc

# ========================================================================
# --- EVALUATION RUN ---
# ========================================================================
if __name__ == "__main__":
    best_dnum = 1
    best_gamma = 5

    print("\n" + "="*80)
    print(f" SECTION VI: ADAPTIVE COMPILER TRACE (LoLa-MNIST | dnum={best_dnum}, gamma={best_gamma})")
    print("="*80)
    sim_lola('hybrid', best_dnum, best_gamma, trace=True)
    print("="*80 + "\n")

    h = sim_lola('hks', best_dnum, best_gamma)
    k = sim_lola('klss', best_dnum, best_gamma)
    hyb = sim_lola('hybrid', best_dnum, best_gamma)

    print("="*50)
    print(f" FINAL LATENCY RESULTS (dnum={best_dnum}, gamma={best_gamma})")
    print("="*50)
    print(f"HKS-Only Total Cycles:      {h:,.0f}")
    print(f"KLSS-Only Total Cycles:     {k:,.0f}")
    print(f"Hybrid (Adaptive) Cycles:   {hyb:,.0f}")
    print("-" * 50)
    print(f"Hybrid Exec Time (300MHz):  {(hyb / FREQ)*1000:.2f} ms")
    print(f"Hybrid Speedup vs KLSS:     {k/hyb:.3f}x")
    print(f"Hybrid Speedup vs HKS:      {h/hyb:.3f}x")
    print("="*50 + "\n")

    labels = ['HKS-Only\n(Static)', 'KLSS-Only\n(Static)', 'Hybrid\n(Adaptive)']
    cycles = [h, k, hyb]
    colors = ['royalblue', 'darkorange', 'forestgreen']

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, cycles, color=colors, edgecolor='black', zorder=3, width=0.6)
    plt.ylabel('Total Execution Latency (Cycles)', fontsize=17, fontweight='bold')
    plt.title(f'LoLa-MNIST Execution Latency\n', fontsize=20, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Give the plot some extra headroom so the double-line text doesn't hit the ceiling
    plt.ylim(0, max(cycles) * 1.15)

    for bar, cyc in zip(bars, cycles):
        # Calculate the actual time in seconds based on the accelerator frequency
        time_secs = (cyc / FREQ)*1000

        # Format the label with cycles on top and seconds on the bottom
        label_text = f'{cyc:,.0f}\n({time_secs:.3f} msecs)'

        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(cycles)*0.015),
                 label_text, ha='center', va='bottom', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()
