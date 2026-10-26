import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ========================================================================
# --- SOTA HARDWARE PARAMETERS (Alveo U280 Normalized) ---
# ========================================================================
NUM_ALU = 512
PERM_TPUT = 512
BW_MAX = 460e9
FREQ = 300e6
N = 2**16
SRAM_CAP = 43e6

MOD_Q_BITS = 1326
MOD_P_BITS = 510
L_MAX = 38

# ========================================================================
# --- SPU-OPTIMIZED LATENCY PRIMITIVES ---
# ========================================================================
def mem_cycles(bytes_rw):
    return (bytes_rw / BW_MAX) * FREQ

def bottleneck(compute, bytes_rw):
    return max(compute, mem_cycles(bytes_rw))

def get_ntt(num_limbs, bytes_per_coefficient, parallelism, in_place):
    t_compute = num_limbs * math.log2(N) * (N / (2 * NUM_ALU * parallelism))
    working_set = num_limbs * N * bytes_per_coefficient
    bytes_rw = 0
    if working_set > SRAM_CAP:
        bytes_rw = working_set if in_place else 2 * working_set
    return bottleneck(t_compute, bytes_rw)

def get_bconv(l_in, l_out, num_digits, bytes_per_coefficient, parallelism, fused=False):
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

def get_mod(num_ops, bytes_per_coefficient, parallelism, evk_bytes=0):
    t_compute = num_ops / (NUM_ALU * parallelism)
    working_set = num_ops * bytes_per_coefficient
    bytes_rw = evk_bytes
    if working_set > SRAM_CAP:
        bytes_rw += working_set
    return bottleneck(t_compute, bytes_rw)

# ========================================================================
# --- KSK OPERATIONS (Double-Hoisting Split Math) ---
# ========================================================================
def hks_modup(logical_l, dnum, k):
    p, b = 2.0, 34/8
    cap_l = math.ceil(MOD_Q_BITS / 34) - 1
    l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)

    al = math.ceil((cap_l + 1) / dnum)
    be = math.ceil((l + 1) / al)

    intt_modup = be * get_ntt(al, b, p, 1)
    bconv_modup = get_bconv(al, al*(be-1)+k, be, b, p, True)
    ntt_modup = be * get_ntt(al*(be-1)+k, b, p, 1)
    return intt_modup + bconv_modup + ntt_modup

def hks_inner_product(logical_l, dnum, k):
    p, b = 2.0, 34/8
    cap_l = math.ceil(MOD_Q_BITS / 34) - 1
    l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)

    al = math.ceil((cap_l + 1) / dnum)
    be = math.ceil((l + 1) / al)

    evk_bytes = be * 2 * (l + 1 + k) * N * b
    mult = get_mod(2*be*(l+1+k)*N, b, p, evk_bytes)
    add = get_mod(2*(be-1)*(l+1+k)*N, b, p)
    return mult + add

def hks_mod_down(logical_l, dnum, k):
    p, b = 2.0, 34/8
    l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)

    intt_moddown = 2 * get_ntt(k, b, p, 1)
    bconv_moddown = get_bconv(k, l+1, 2, b, p, True)
    ntt_moddown = 2 * get_ntt(l+1, b, p, 0)
    mod_sub = get_mod(2*(l+1)*N, b, p)
    return intt_moddown + bconv_moddown + ntt_moddown + mod_sub

def klss_modup(logical_l, dnum, k, v, atilde, aprime):
    p, bs, bl = 1.0, 34/8, 62/8
    cap_l = math.ceil(MOD_Q_BITS / 34) - 1
    l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)

    al = math.ceil((cap_l+1)/dnum)
    be = math.ceil((l + 1) / al)

    intt_modup = be * get_ntt(al, bs, p, 1)
    bconv_modup = get_bconv(al, aprime, be, bl, p, True)
    ntt_modup = be * get_ntt(aprime, bl, p, 1)
    return intt_modup + bconv_modup + ntt_modup

def klss_inner_product(logical_l, dnum, k, gamma, v, atilde, aprime):
    p, bs, bl = 1.0, 34/8, 62/8
    cap_l = math.ceil(MOD_Q_BITS / 34) - 1
    l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)

    al = math.ceil((cap_l+1)/dnum)
    be = math.ceil((l + 1) / al)
    beta_t = math.ceil((l + 1 + al) / gamma)
    beta_t_static = math.ceil((cap_l + 1 + al) / gamma)

    evk_bytes = be * beta_t_static * 2 * aprime * N * bl
    mult = get_mod(2 * aprime * be * beta_t * N, bl, p, evk_bytes)
    add = get_mod(2 * aprime * (be - 1) * beta_t * N, bl, p)
    return mult + add

def klss_mod_down(logical_l, dnum, k, gamma, v, atilde, aprime):
    p, bs, bl = 1.0, 34/8, 62/8
    cap_l = math.ceil(MOD_Q_BITS / 34) - 1
    l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)

    al = math.ceil((cap_l+1)/dnum)
    l_tilde = l + 1 + al
    beta_t = math.ceil(l_tilde / gamma)
    chunk_size = math.ceil(l_tilde / beta_t)

    # FIX: Correctly scale KLSS ModDown by 2 * beta_t to model recovery penalty
    intt_moddown = 2 * beta_t * get_ntt(aprime, bl, p, 1)
    bconv_moddown_1 = get_bconv(aprime, chunk_size, 2 * beta_t, bl, p, True)
    bconv_moddown_2 = get_bconv(al, l+1, 2, bl, p, True)
    ntt_moddown = 2 * get_ntt(l+1, bs, p, 0)
    mod_sub = get_mod(2 * (l+1) * N, bs, p)

    return intt_moddown + bconv_moddown_1 + bconv_moddown_2 + ntt_moddown + mod_sub

# ========================================================================
# --- DYNAMIC COMPILER LOGIC (DOUBLE-HOISTING IMPLEMENTED) ---
# ========================================================================
def get_hoisted(logical_l, r, mode, dnum, k=9, gamma=7, v=60, atilde=6, aprime=2, trace_info=""):
    phys_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)

    # Double Hoisting: ModUp and ModDown outside the r loop!
    h_ip = hks_inner_product(logical_l, dnum, k)
    h_md = hks_mod_down(logical_l, dnum, k)
    h_c = hks_modup(logical_l, dnum, k) + r * (2*(phys_l+1+k)*(N/PERM_TPUT) + h_ip) + h_md

    k_ip = klss_inner_product(logical_l, dnum, k, gamma, v, atilde, aprime)
    k_md = klss_mod_down(logical_l, dnum, k, gamma, v, atilde, aprime)
    k_c = klss_modup(logical_l, dnum, k, v, atilde, aprime) + r * (2*aprime*(N/PERM_TPUT) + k_ip) + k_md

    picked = "HKS " if h_c < k_c else "KLSS"
    if mode == 'hybrid' and trace_info:
        print(f"  [L={logical_l:02d} | {trace_info:<13} | r={r:02d}] HKS: {h_c:,.0f} | KLSS: {k_c:,.0f}  ->  {picked} wins")

    return (h_c if mode == 'hks' else k_c if mode == 'klss' else min(h_c, k_c)) + r * get_mod(2*(phys_l+1)*N, 34/8, 2.0)

def get_cmult(logical_l, mode, dnum, k=9, gamma=7, v=60, atilde=6, aprime=2, trace_info=""):
    phys_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)

    b = 4 * get_mod(2*(phys_l+1)*N, 34/8, 2.0)

    h_c = hks_modup(logical_l, dnum, k) + hks_inner_product(logical_l, dnum, k) + hks_mod_down(logical_l, dnum, k)
    k_c = klss_modup(logical_l, dnum, k, v, atilde, aprime) + klss_inner_product(logical_l, dnum, k, gamma, v, atilde, aprime) + klss_mod_down(logical_l, dnum, k, gamma, v, atilde, aprime)

    picked = "HKS " if h_c < k_c else "KLSS"
    if mode == 'hybrid' and trace_info:
        print(f"  [L={logical_l:02d} | {trace_info:<13} | CMult] HKS: {h_c:,.0f} | KLSS: {k_c:,.0f}  ->  {picked} wins")

    return b + (h_c if mode == 'hks' else k_c if mode == 'klss' else min(h_c, k_c))

# ========================================================================
# --- BOOTSTRAPPING PIPELINE TRACE (Advanced Bootstrapping Algorithm) ---
# ========================================================================
def sim_bootstrapping_phases(mode, dnum=3, gamma=7, aprime=2, atilde=6, stc_lvl=5, pe_lvl=5, cts_lvl=5, r_stc=3, r_cts=8):
    k_val = 15
    cap_l = math.ceil(MOD_Q_BITS / 34) - 1

    if mode == 'hybrid':
        print(f"\n{'='*80}\n COMPILER TRACE (Bootstrapping | dnum={dnum}, gamma={gamma})\n{'='*80}")

    phases = {'StC': 0, 'ModUp': 0, 'CtS': 0, 'PolyEval': 0}

    # 1. StC Phase (Shallow Levels: Starts at stc_lvl, goes down to 0)
    logical_l = stc_lvl
    for _ in range(stc_lvl):
        ti = "StC Rotations" if mode == 'hybrid' else ""
        phys_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)
        phases['StC'] += get_hoisted(logical_l, r_stc, mode, dnum, k=k_val, gamma=gamma, atilde=atilde, aprime=aprime, trace_info=ti) * 2 + 36*get_mod(2*(phys_l+1)*N, 34/8, 2.0) + get_mod(2*phys_l*N, 34/8, 2.0)
        logical_l -= 1

    # 2. ModUp Phase (ModRaise BConv Included! Raises from phys_l=0 to cap_l)
    phys_l = 0 # At the absolute bottom of the circuit
    bconv_modraise = get_bconv(phys_l, cap_l, 2, 34/8, 2.0, fused=True)
    ntt_modraise = cap_l * get_ntt(cap_l, 34/8, 1.0, 1)
    phases['ModUp'] += bconv_modraise + ntt_modraise

    # Ciphertext is refreshed!
    logical_l = L_MAX
    if mode == 'hybrid': print("-" * 80)

    # 3. CtS Phase (Deep Levels: Starts at L_MAX, goes down)
    for _ in range(cts_lvl):
        ti = "CtS Rotations" if mode == 'hybrid' else ""
        phys_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)
        phases['CtS'] += get_hoisted(logical_l, r_cts, mode, dnum, k=k_val, gamma=gamma, atilde=atilde, aprime=aprime, trace_info=ti) * 2 + 36*get_mod(2*(phys_l+1)*N, 34/8, 2.0) + get_mod(2*phys_l*N, 34/8, 2.0)
        logical_l -= 1

    if mode == 'hybrid': print("-" * 80)

    # 4. PolyEval Phase (2x Multiplier for Real/Complex Slots)
    total_bsgs_mults = 8
    cmults_base_per_level = total_bsgs_mults // pe_lvl
    cmults_remainder = total_bsgs_mults % pe_lvl

    for idx in range(pe_lvl):
        ti = "PolyEval Math" if mode == 'hybrid' else ""
        phys_l = max(0, math.ceil((logical_l / L_MAX) * MOD_Q_BITS / 34) - 1)

        current_cmults = cmults_base_per_level + (1 if idx < cmults_remainder else 0)
        level_cost = 0

        for m in range(current_cmults):
            level_cost += get_cmult(logical_l, mode, dnum, k=k_val, gamma=gamma, atilde=atilde, aprime=aprime, trace_info=ti if m == 0 else "") + get_mod(2*(phys_l+1)*N, 34/8, 2.0)

        level_cost += get_mod(2*phys_l*N, 34/8, 2.0)

        # Multiply the entire BSGS level's execution cost by 2 (Real + Imaginary)
        phases['PolyEval'] += level_cost * 2
        logical_l -= 1

    if mode == 'hybrid': print("="*80 + "\n")

    return {k: (v / FREQ) * 1000 for k, v in phases.items()}

# ========================================================================
# --- EVALUATION RUN & VISUALIZATION ---
# ========================================================================
if __name__ == "__main__":
    dnum_opt = 3
    # gamma_opt = 7
    aprime_opt = 2
    gamma_opt = 9

    h_p = sim_bootstrapping_phases('hks', dnum=dnum_opt, gamma=gamma_opt, aprime=aprime_opt)
    k_p = sim_bootstrapping_phases('klss', dnum=dnum_opt, gamma=gamma_opt, aprime=aprime_opt)
    hyb_p = sim_bootstrapping_phases('hybrid', dnum=dnum_opt, gamma=gamma_opt, aprime=aprime_opt)

    h_total = sum(h_p.values())
    k_total = sum(k_p.values())
    hyb_total = sum(hyb_p.values())

    print(f"TOTAL HKS:    {h_total:.2f} ms")
    print(f"TOTAL KLSS:   {k_total:.2f} ms")
    print(f"TOTAL HYBRID: {hyb_total:.2f} ms")
    print("-" * 50)
    print(f"Hybrid Speedup vs KLSS:     {k_total / hyb_total:.3f}x")
    print(f"Hybrid Speedup vs HKS:      {h_total / hyb_total:.3f}x")
    print("=" * 50 + "\n")

    # Visualization Generation
    labels = ['Static HKS', 'Static KLSS', 'Adaptive Hybrid']
    stc_data = [h_p['StC'], k_p['StC'], hyb_p['StC']]
    modup_data = [h_p['ModUp'], k_p['ModUp'], hyb_p['ModUp']]
    cts_data = [h_p['CtS'], k_p['CtS'], hyb_p['CtS']]
    pe_data = [h_p['PolyEval'], k_p['PolyEval'], hyb_p['PolyEval']]

    fig, ax = plt.subplots(figsize=(9, 6))
    width = 0.55

    p1 = ax.bar(labels, stc_data, width, label='StC (Slot-to-Coeff)', color='#4c72b0', edgecolor='black')
    p2 = ax.bar(labels, modup_data, width, bottom=stc_data, label='ModUp (ModRaise)', color='#dd8452', edgecolor='black')
    p3 = ax.bar(labels, cts_data, width, bottom=np.add(stc_data, modup_data), label='CtS (Coeff-to-Slot)', color='#55a868', edgecolor='black')
    p4 = ax.bar(labels, pe_data, width, bottom=np.add(np.add(stc_data, modup_data), cts_data), label='PolyEval (BSGS 31-Deg x2)', color='#c44e52', edgecolor='black')

    totals = [h_total, k_total, hyb_total]
    for i, total in enumerate(totals):
        ax.text(i, total + 1.5, f"{total:.2f} ms", ha='center', fontweight='bold', fontsize=13)

    ax.set_ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title(f'Bootstrapping 4-Phase Breakdown ($L_{{boot}} = 15$)', fontsize=17, fontweight='bold')

    handles, legend_labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], legend_labels[::-1], loc='upper right', fontsize=13)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # SOTA Comparison Bar Chart
    sota_labels = ['FAB (Norm.)\n(2022)', 'Poseidon (Norm.)\n(2023)', 'FAST\n(2025)', 'HERA\n(2026)', 'Hybrid\n(Ours)']
    sota_latencies = [92.4, 75.4, 105.2, 40.2, hyb_total]

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    bars = ax2.bar(sota_labels, sota_latencies, color='lightgray', edgecolor='black', width=0.6)
    bars[3].set_color('salmon')
    bars[3].set_edgecolor('black')
    bars[3].set_hatch('///')
    bars[4].set_color('forestgreen')
    bars[4].set_edgecolor('black')

    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f} ms", ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Normalized Bootstrapping Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Bootstrapping Latency vs. SOTA Accelerators\n(Standardized to 1024 ALUs @ 300MHz)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='black', label='Normalized Prior Art'),
        Patch(facecolor='salmon', edgecolor='black', hatch='///', label='HERA (Static)'),
        Patch(facecolor='forestgreen', edgecolor='black', label='Ours (Adaptive Hybrid)')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()
