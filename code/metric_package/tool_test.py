import time
import torch
import os
from typing import Callable, Optional, Dict, Any
from metric_package.statistics_computation_spd import (
    SPDDataBundle,
    stack_spd_triplet,
    slice_spd_bundle,
    emdf_product_pair_batch,
    f_perp_gen_pair_batch,
    generate_data_spd,
    Bundle_spd,
)
from metric_package.generators_spd import (
    generate_conditional_samples,
    FittedConditionalGenerators,
    FlowTrainConfig,
    train_val_split_spd_triplet,
    make_generator,
    OracleGenerators,
)


def profile_function(
    func: Callable[..., Any],
    func_kwargs: Optional[Dict] = None,
    device: torch.device | None = None,
    sync_cuda: bool = True,
    clear_cache: bool = True,
) -> Dict[str, Any]:
    """
    Profile a function using func + kwargs style.

    Parameters
    ----------
    func : Callable
        Function to run.

    func_kwargs : dict or None
        Keyword arguments passed to func.

    device : torch.device or None
        CUDA device for profiling.

    Returns
    -------
    dict
    """

    if func_kwargs is None:
        func_kwargs = {}

    use_cuda = (
        device is not None
        and device.type == "cuda"
        and torch.cuda.is_available()
    )

    if use_cuda:
        torch.cuda.set_device(device)

        if clear_cache:
            torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()

        if sync_cuda:
            torch.cuda.synchronize()

        total_mem = torch.cuda.get_device_properties(device).total_memory
    else:
        total_mem = None

    # -------------------------
    # timing
    # -------------------------
    t0 = time.perf_counter()

    result = func(**func_kwargs)

    if use_cuda and sync_cuda:
        torch.cuda.synchronize()

    t1 = time.perf_counter()

    # -------------------------
    # GPU memory
    # -------------------------
    if use_cuda:
        max_alloc = torch.cuda.max_memory_allocated(device)
        max_reserved = torch.cuda.max_memory_reserved(device)

        max_alloc_mb = max_alloc / 1024**2
        max_reserved_mb = max_reserved / 1024**2

        max_alloc_pct = max_alloc / total_mem * 100
        max_reserved_pct = max_reserved / total_mem * 100
    else:
        max_alloc_mb = None
        max_reserved_mb = None
        max_alloc_pct = None
        max_reserved_pct = None

    return {
        "result": result,
        "elapsed_sec": t1 - t0,

        "gpu_max_allocated_mb": max_alloc_mb,
        "gpu_max_reserved_mb": max_reserved_mb,

        "gpu_max_allocated_pct": max_alloc_pct,
        "gpu_max_reserved_pct": max_reserved_pct,
    }


def statistics_test(
    Bundle_X: SPDDataBundle,
    Bundle_Y: SPDDataBundle,
    Bundle_Z: SPDDataBundle,
    Bundle_X_orc: SPDDataBundle,
    Bundle_Y_orc: SPDDataBundle,
    atol: float = 1e-12,
    batch_size: int = 1024,
    chunk_size_xy: int | None = None,
    chunk_size_z: int | None = None,
    sync_cuda: bool = True,
) -> Dict[str, Any]:
    """
    Test/profiling version of statistics.

    Outputs:
    - statistic value
    - per-batch emdf_P time
    - per-batch emdf_I time
    - total emdf_P time
    - total emdf_I time
    - main loop time
    - whole function time
    """

    def _sync():
        if sync_cuda and Bundle_X.matrix.is_cuda:
            torch.cuda.synchronize(Bundle_X.matrix.device)

    _sync()
    t_func_start = time.perf_counter()

    # if Bundle_X.matrix.shape != Bundle_Y.matrix.shape or Bundle_X.matrix.shape != Bundle_Z.matrix.shape:
    #     raise ValueError("Bundle_X, Bundle_Y, and Bundle_Z must have identical shapes.")

    # if Bundle_X.matrix.ndim != 3:
    #     raise ValueError(
    #         "Bundle_X.matrix, Bundle_Y.matrix, Bundle_Z.matrix must have shape (n, p, p)."
    #     )

    # if Bundle_X_orc.matrix.ndim != 4 or Bundle_Y_orc.matrix.ndim != 4:
    #     raise ValueError(
    #         "Bundle_X_orc.matrix and Bundle_Y_orc.matrix must have shape (n, M, p, p)."
    #     )

    n = Bundle_X.matrix.shape[0]
    # if n < 2:
    #     raise ValueError("At least two observations are required.")

    # if Bundle_X_orc.matrix.shape[0] != n or Bundle_Y_orc.matrix.shape[0] != n:
    #     raise ValueError(
    #         "Bundle_X_orc and Bundle_Y_orc must have leading dimension n matching observed data."
    #     )

    Bundle_S = stack_spd_triplet(Bundle_X, Bundle_Y, Bundle_Z)

    total_pairs = n * (n - 1)
    total = 0.0
    device = Bundle_X.matrix.device

    emdf_P_times = []
    emdf_I_times = []
    batch_sizes = []

    _sync()
    t_loop_start = time.perf_counter()

    for start in range(0, total_pairs, batch_size):
        end = min(start + batch_size, total_pairs)
        cur_batch_size = end - start
        batch_sizes.append(cur_batch_size)

        k = torch.arange(start, end, device=device, dtype=torch.long)
        ib = torch.div(k, n - 1, rounding_mode="floor")
        r = torch.remainder(k, n - 1)
        jb = r + (r >= ib).to(torch.long)

        Bundle_u_batch = slice_spd_bundle(Bundle_S, ib)
        Bundle_v_batch = slice_spd_bundle(Bundle_S, jb)

        # -----------------------------
        # timing emdf_P
        # -----------------------------
        _sync()
        t0 = time.perf_counter()

        emdf_P = emdf_product_pair_batch(
            Bundle_xyz_samples=Bundle_S,
            Bundle_u_batch=Bundle_u_batch,
            Bundle_v_batch=Bundle_v_batch,
            atol=atol,
        )

        _sync()
        t1 = time.perf_counter()
        emdf_P_times.append(t1 - t0)

        # -----------------------------
        # timing emdf_I
        # -----------------------------
        _sync()
        t2 = time.perf_counter()

        emdf_I = f_perp_gen_pair_batch(
            Bundle_u_batch=Bundle_u_batch,
            Bundle_v_batch=Bundle_v_batch,
            Bundle_X_orc=Bundle_X_orc,
            Bundle_Y_orc=Bundle_Y_orc,
            Bundle_Z=Bundle_Z,
            atol=atol,
            chunk_size_xy=chunk_size_xy,
            chunk_size_z=chunk_size_z,
        )

        _sync()
        t3 = time.perf_counter()
        emdf_I_times.append(t3 - t2)

        diff = emdf_P - emdf_I
        total += float((diff * diff).sum().item())

    _sync()
    t_loop_end = time.perf_counter()

    stat_value = total / total_pairs

    _sync()
    t_func_end = time.perf_counter()

    emdf_P_total = float(sum(emdf_P_times))
    emdf_I_total = float(sum(emdf_I_times))

    return {
        "statistic": stat_value,

        "total_elapsed_sec": t_func_end - t_func_start,
        "loop_elapsed_sec": t_loop_end - t_loop_start,

        "emdf_P_total_sec": emdf_P_total,
        "emdf_I_total_sec": emdf_I_total,

        "emdf_P_avg_sec": emdf_P_total / len(emdf_P_times),
        "emdf_I_avg_sec": emdf_I_total / len(emdf_I_times),

        "emdf_P_first_sec": emdf_P_times[0],
        "emdf_I_first_sec": emdf_I_times[0],

        "emdf_P_times": emdf_P_times,
        "emdf_I_times": emdf_I_times,
        "batch_sizes": batch_sizes,

        "num_batches": len(batch_sizes),
        "total_pairs": total_pairs,
        "batch_size": batch_size,
    }


def run_spd_flow_debug(
    BASE_SEED: int = 20260000 + 1000 * 1 + 16,
    USE_GPU: bool = True,
    GPU_ID: int = 1,

    N_SAMPLES: int = 150,
    M: int = 75,
    SIZE: int = 2,
    RHO: float = 0.0,
    BATCH_SIZE: int = 1024,
    ORC: bool = True,

    # FlowTrainConfig 参数
    epochs: int = 500,
    flow_batch_size: int = 256 * 2,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    grad_clip_norm: float | None = 4.0,
    scheduler_gamma: float = 0.96,
    verbose: bool = True,
    patience: int | None = None,

    # FittedConditionalGenerators.fit 参数
    hidden_dim: int = 128 * 2,
    num_hidden_layers: int = 2,
    scale_limit: float = 4.0,
    dropout: float = 0.15,

    val_ratio: float = 0.25,

    # info
    info = False,
):
    # =========================
    # GPU settings
    # =========================
    if USE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

    CONFIG = FlowTrainConfig(
        epochs=epochs,
        batch_size=flow_batch_size,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip_norm=grad_clip_norm,
        scheduler_gamma=scheduler_gamma,
        verbose=verbose,
        patience=patience,
    )

    GPU = USE_GPU and torch.cuda.is_available()
    device = torch.device("cuda" if GPU else "cpu")

    if info:
        print("===== Device Info =====")
        print(f"USE_GPU requested        : {USE_GPU}")
        print(f"CUDA available           : {torch.cuda.is_available()}")
        print(f"GPU flag used            : {GPU}")
        print(f"GPU_ID                   : {GPU_ID}")
        print(f"BATCH_SIZE               : {BATCH_SIZE}")

        if GPU:
            print(f"Visible CUDA device(s)   : {os.environ.get('CUDA_VISIBLE_DEVICES')}")
            print(f"torch current device     : {torch.cuda.current_device()}")
            print(f"torch device name        : {torch.cuda.get_device_name(torch.cuda.current_device())}")

        print("\n===== Flow Config =====")
        print(CONFIG)
        print(f"hidden_dim               : {hidden_dim}")
        print(f"num_hidden_layers        : {num_hidden_layers}")
        print(f"scale_limit              : {scale_limit}")
        print(f"dropout                  : {dropout}")
        print()

    # =========================
    # Total timing
    # =========================
    print("===== Timing Breakdown =====")
    t_total_start = time.time()

    # =========================
    # Step 1: generate data
    # =========================
    t0 = time.time()

    X, Y, Z = generate_data_spd(
        n=N_SAMPLES,
        size=SIZE,
        rho=RHO,
        seed=BASE_SEED,
    )

    Bundle_X = Bundle_spd(X)
    Bundle_Y = Bundle_spd(Y)
    Bundle_Z = Bundle_spd(Z)

    t1 = time.time()
    if info: 
        print(f"Data generation time      : {t1 - t0:.4f} seconds", flush=True)

    # =========================
    # Step 2: train/val split
    # =========================
    (
        Bundle_X_train,
        Bundle_Y_train,
        Bundle_Z_train,
        Bundle_X_val,
        Bundle_Y_val,
        Bundle_Z_val,
    ) = train_val_split_spd_triplet(
        Bundle_X,
        Bundle_Y,
        Bundle_Z,
        val_ratio=val_ratio,
        seed=BASE_SEED + 17,
    )

    # =========================
    # Step 3: fit conditional generators
    # =========================
    t_fit_start = time.time()

    fitted_generators, fit_history = FittedConditionalGenerators.fit(
        Bundle_X=Bundle_X_train,
        Bundle_Y=Bundle_Y_train,
        Bundle_Z=Bundle_Z_train,
        config=CONFIG,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        scale_limit=scale_limit,
        dropout=dropout,
        Bundle_X_val=Bundle_X_val,
        Bundle_Y_val=Bundle_Y_val,
        Bundle_Z_val=Bundle_Z_val,
    )

    t_fit_end = time.time()
    x_train_nll_last = float(fit_history["x_history"]["train_nll"][-1])
    y_train_nll_last = float(fit_history["y_history"]["train_nll"][-1])

    if info:
        print(f"Flow fitting time         : {t_fit_end - t_fit_start:.4f} seconds", flush=True)

        
        print(f"x_train_nll_last         : {x_train_nll_last:.6f}", flush=True)
        print(f"y_train_nll_last         : {y_train_nll_last:.6f}", flush=True)

    # =========================
    # Step 4: generate fitted samples
    # =========================
    t_gen_start = time.time()

    gen = make_generator(Bundle_Z.matrix.device, seed=BASE_SEED + 29)

    Bundle_X_gen, Bundle_Y_gen = generate_conditional_samples(
        Bundle_Z=Bundle_Z,
        M=M,
        generators=fitted_generators,
        generator=gen,
    )

    Bundle_X_orc = None
    Bundle_Y_orc = None

    if ORC:
        oracle_generators = OracleGenerators(sigma_perm=2.0)

        gen_orc = make_generator(device=device, seed=BASE_SEED + 29)

        Bundle_X_orc, Bundle_Y_orc = generate_conditional_samples(
            Bundle_Z=Bundle_Z,
            M=M,
            generators=oracle_generators,
            generator=gen_orc,
            chunk_size=1024,
        )

    t_gen_end = time.time()
    if info:
        
        print(f"Conditional sampling time : {t_gen_end - t_gen_start:.4f} seconds", flush=True)

    if GPU:
        torch.cuda.empty_cache()

    # =========================
    # Step 5: compute statistics
    # =========================
    t_stat_start = time.time()

    T_gen = statistics_test(
        Bundle_X,
        Bundle_Y,
        Bundle_Z,
        Bundle_X_gen,
        Bundle_Y_gen,
        batch_size=BATCH_SIZE,
    )

    T_orc = None

    if ORC:
        T_orc = statistics_test(
            Bundle_X,
            Bundle_Y,
            Bundle_Z,
            Bundle_X_orc,
            Bundle_Y_orc,
            batch_size=BATCH_SIZE,
        )

    t_stat_end = time.time()
    if info:
        print(f"Statistics time           : {t_stat_end - t_stat_start:.4f} seconds", flush=True)

    # =========================
    # Total
    # =========================
    
    t_total_end = time.time()

    print("---------------------------------", flush=True)
    print(f"Total time                : {t_total_end - t_total_start:.4f} seconds", flush=True)

    print("\n===== Test Result =====", flush=True)
    print(f"T_obs_gen                 : {T_gen['statistic']}", flush=True)

    if ORC:
        print(f"T_obs_orc                 : {T_orc['statistic']}", flush=True)

    if GPU and info:
        print("\n===== GPU Memory Summary =====", flush=True)
        print(
            f"Max memory allocated      : "
            f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MB",
            flush=True,
        )
        print(
            f"Max memory reserved       : "
            f"{torch.cuda.max_memory_reserved() / 1024**2:.2f} MB",
            flush=True,
        )

    return {
        "T_gen": T_gen,
        "T_orc": T_orc,
        "fit_history": fit_history,
        "fitted_generators": fitted_generators,
        "x_train_nll_last": x_train_nll_last,
        "y_train_nll_last": y_train_nll_last,
        "timing": {
            "data_generation": t1 - t0,
            "flow_fitting": t_fit_end - t_fit_start,
            "conditional_sampling": t_gen_end - t_gen_start,
            "statistics": t_stat_end - t_stat_start,
            "total": t_total_end - t_total_start,
        },
    }