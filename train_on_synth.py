import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import random
import pyroomacoustics as pra

# ── 1. Pyroomacoustics Dataset Generator ───────────────────────────────────────
def generate_pra_dataset(num_rooms=10, rate=48_000, length_samples=48_000):
    print(f"Generating realistic PRA dataset of {num_rooms} rooms...")
    rirs = []
    for i in range(num_rooms):
        dim = np.random.uniform([3.0, 3.0, 2.5], [10.0, 8.0, 5.0])
        rt60_tgt = np.random.uniform(0.4, 1.5)
        e_absorption, max_order = pra.inverse_sabine(rt60_tgt, dim)
        room = pra.ShoeBox(dim, fs=rate, materials=pra.Material(e_absorption), max_order=max_order)
        source_pos = np.random.uniform([0.5, 0.5, 0.5], dim - 0.5)
        room.add_source(source_pos)
        mic_pos = np.random.uniform([0.5, 0.5, 0.5], dim - 0.5)
        room.add_microphone(mic_pos)
        room.compute_rir()
        rir = room.rir[0][0]
        rir = np.abs(rir)
        if np.max(rir) > 0:
            rir /= np.max(rir)
        if len(rir) < length_samples:
            rir = np.pad(rir, (0, length_samples - len(rir)))
        else:
            rir = rir[:length_samples]
        rirs.append(rir)
    return rirs

# ── 2. Your FDN Algorithm (Updated to return metrics) ──────────────────────
def fit_fdn_to_rir(rir, rate=48_000, epoch_count=5000):
    spm = rate // 1000
    n = len(rir)
    
    sorted_idx = np.argsort(rir)[::-1]
    scale_index = sorted_idx[43]

    # ── Target Acoustic Descriptors
    energy = np.linalg.norm(rir)**2
    energy_50ms = np.linalg.norm(rir[: 50 * spm])**2
    energy_80ms = np.linalg.norm(rir[: 80 * spm])**2
    early_energy = np.linalg.norm(rir[:scale_index-1])**2

    target_t30_idx = 0
    for i in range(n - 1, -1, -1):
        if rir[i] >= 0.001:
            target_t30_idx = i
            break
            
    target_metrics = {
        "t30 (samples)": target_t30_idx,
        "clarity_db": 10 * np.log10(energy_80ms / energy),
        "definition": energy_50ms / energy,
        "center_time (samples)": np.sum((rir ** 2) * np.arange(n)) / energy
    }

    # ── FDN architecture setup
    k = 16
    alpha = 0.99
    locs = np.array([0, 4, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 2560, 2816, 3072, 3584, 4096], dtype=np.float32).reshape(k, 1)
    # locs[-1] = max(locs[-1], target_t30_idx-spm*15)

    upper_bound = target_t30_idx - spm * 15
    locs_normalized = locs / locs.max()
    locs = locs_normalized * upper_bound
    locs = locs.astype(np.int32)

    coeffs = np.ones(2 * k, dtype=np.float32) * alpha
    a_weights = jnp.array(coeffs)
    b_decay = jnp.array(coeffs)

    times_np = np.arange(n, dtype=np.float32).reshape(1, n).repeat(k, axis=0) - (locs - 1)
    activation_np = np.clip(times_np, 0.0, 1.0)
    times = jnp.array(times_np - 1, dtype=jnp.float32)
    activation = jnp.array(activation_np, dtype=jnp.float32)

    @jit
    def synthesize(params):
        safe_times = jnp.maximum(times, 0.0)
        weights = params[:k].reshape(k, 1)
        decay_bases = params[k:].reshape(k, 1)
        decay = jnp.pow(decay_bases, safe_times)
        y = activation * decay * (weights ** 2)
        
        return jnp.sum(y, axis=0)

    @jit
    def compute_losses(params):
        y = synthesize(params)
        
        # Fixed total energy to correctly omit the trailing samples mirroring early reflection count
        eng = early_energy + jnp.linalg.norm(y[:-scale_index-1])**2
        eng_80 = early_energy + jnp.linalg.norm(y[: 80 * spm - scale_index-1])**2
        eng_50 = early_energy + jnp.linalg.norm(y[: 50 * spm - scale_index-1])**2
        
        definition = eng_50 / eng
        simp_clarity = eng_80 / eng
        center_time = jnp.sum(y * y * jnp.arange(n)) / eng

        l_def = (target_metrics["definition"] - definition)**2
        l_cl  = ((energy_80ms / energy) - simp_clarity)**2
        l_ct  = (target_metrics["center_time (samples)"] - center_time)**2
        l_t30 = (y[target_t30_idx - scale_index-1] - 0.001)**2
        
        return 1e4 * l_def + 1e7 * l_cl + 2e-6 * l_ct

    grad_loss = jit(grad(compute_losses))
    
    # Training Loop
    curr_a = a_weights
    lr_base = 5e-8
    lr_schedule = [
        lr_base,        lr_base,
        lr_base * 10,   lr_base * 10,
        lr_base * 50,   lr_base * 100,
        lr_base * 50,   lr_base * 50,
        lr_base * 10,   lr_base,
        lr_base * 0.1,
    ]
    lr_interval = epoch_count // 10
    Lalpha = 1e-5
    lr_idx = 0
    for epoch in range(epoch_count + 1):
        multipliers = jnp.where(jnp.arange(curr_a.shape[0]) < k, 1.0, Lalpha)
        curr_a = curr_a - grad_loss(curr_a) * lr_schedule[lr_idx] * multipliers
        if epoch % lr_interval == 0:
            lr_idx = min(lr_idx + 1, len(lr_schedule) - 1)

        if epoch % lr_interval == 0:
            total= compute_losses(curr_a)

            print(f"\n── Epoch {epoch:,} │ lr = {lr_schedule[lr_idx]:.2e} │ total loss = {float(total):.6f}")


    # ── Final Estimation Calculation
    final_y = np.array(synthesize(curr_a))
    print(curr_a)
    
    # Fixed slicing bounds and variables
    est_eng = early_energy + np.linalg.norm(final_y[:-scale_index-1])**2
    est_eng_80 = early_energy + np.linalg.norm(final_y[: 80 * spm - scale_index-1])**2
    est_eng_50 = early_energy + np.linalg.norm(final_y[: 50 * spm - scale_index-1])**2

    # Fixed: No longer adding `scale_index` to the output T30
    est_t30_idx = 0
    for i in range(len(final_y)-1, -1, -1):
        if final_y[i] >= 0.001:
            est_t30_idx = i
            break

    # Fixed formulations for final metrics
    est_metrics = {
        "t30 (samples)": est_t30_idx,
        "clarity_db": 10 * np.log10(est_eng_80 / est_eng),
        "definition": est_eng_50 / est_eng,
        "center_time (samples)": np.sum(final_y * final_y * np.arange(len(final_y))) / est_eng
    }

    return final_y, scale_index, target_metrics, est_metrics

# ── 3. Dataset Processing & Evaluation ─────────────────────────────────────────
if __name__ == "__main__":
    dataset = generate_pra_dataset(num_rooms=100, rate=48_000, length_samples=48_000)
    
    # Dictionary to store the errors across all rooms
    all_errors = {
        "t30 (samples)": [],
        "clarity_db": [],
        "definition": [],
        "center_time (samples)": []
    }
    
    for idx, rir in enumerate(dataset):
        print(f"\n── Training on Room {idx + 1} ──")
        y_synth, s_idx, targets, estimates = fit_fdn_to_rir(rir, epoch_count=500000)
        
        print(f"{'Metric':<22} | {'Target':<12} | {'Estimate':<12}")
        print("-" * 52)
        for key in targets.keys():
            print(f"{key:<22} | {targets[key]:<12.4f} | {estimates[key]:<12.4f}")
            
            # --- CALCULATE RELATIVE ERROR ---
            true_val = targets[key]
            # Use abs() of the difference, then divide by the absolute true value
            # Adding a tiny epsilon (1e-10) prevents division by zero errors
            rel_error = np.abs(estimates[key] - true_val) / (np.abs(true_val) + 1e-10)
            
            all_errors[key].append(rel_error)
            
    # ── 4. Final Aggregated Statistics (Now showing Relative Errors) ──
    print("\n\n" + "=" * 90)
    print("FINAL STATISTICS: RELATIVE ERRORS (abs(est - true) / true)")
    print("=" * 90)
    # Changed header labels to reflect percentages or relative values
    print(f"{'Metric':<22} | {'Mean Rel':<12} | {'Med Rel':<12} | {'Std Dev':<12} | {'Max Rel':<12} | {'Max Abs Rel':<12}")
    print("-" * 90)
    
    for key, err_list in all_errors.items():
        err_arr = np.array(err_list)
        
        print(f"{key:<22} | {np.mean(err_arr):<12.4f} | {np.median(err_arr):<12.4f} | "
              f"{np.std(err_arr):<12.4f} | {np.max(err_arr):<12.4f} | {np.max(np.abs(err_arr)):<12.4f}")
