import heapq
import math
import numpy as np

# ==========================================
# 1. PARAMETER MODEL JOB-SHOP (SEC. 2.7)
# ==========================================
BASE_MACHINES = [3, 2, 4, 3, 1]  # Station 1 s/d 5 (index 0 s/d 4)
JOB_PROBS = [0.3, 0.5, 0.2]
MEAN_INTERARRIVAL = 0.25  # jam

# Routing (0-indexed) dan mean service time (jam) tiap tipe job
ROUTINGS = {
    0: [2, 0, 1, 4],     # Job Type 1: Station 3, 1, 2, 5
    1: [3, 0, 2],        # Job Type 2: Station 4, 1, 3
    2: [1, 4, 0, 3, 2],  # Job Type 3: Station 2, 5, 1, 4, 3
}

SERVICE_MEANS = {
    0: [0.50, 0.60, 0.85, 0.50],
    1: [1.10, 0.80, 0.75],
    2: [1.20, 0.25, 0.70, 0.90, 1.00],
}

WARMUP_HOURS = 10 * 8.0   # 80 jam
DATA_HOURS = 90 * 8.0     # 720 jam
TOTAL_HOURS = WARMUP_HOURS + DATA_HOURS  # 800 jam

# ==========================================
# 2. FUNGSI SIMULASI 1 REPLIKASI JOB-SHOP
# ==========================================
def run_jobshop_replication(machines_per_station, seed):
    rng = np.random.default_rng(seed)
    
    # State antrean dan mesin sibuk per stasiun
    queues = [[] for _ in range(5)]  # elemen: (arrival_to_station_time, job_type, task_idx)
    busy = [0] * 5
    
    # Statistik delay setelah warm-up (per job type)
    total_delay_by_type = [0.0, 0.0, 0.0]
    count_tasks_by_type = [0, 0, 0]
    
    # Event list: (event_time, event_id, event_type, station, job_type, task_idx)
    # event_type: 1 = Arrival to system, 2 = Departure from station, 3 = End simulation
    event_list = []
    event_counter = 0
    
    def schedule_event(time, etype, station=-1, jtype=-1, tidx=-1):
        nonlocal event_counter
        event_counter += 1
        heapq.heappush(event_list, (time, event_counter, etype, station, jtype, tidx))
    
    def gen_erlang2(mean_val):
        # 2-Erlang = jumlah 2 variabel eksponensial dengan mean = mean_val / 2
        return rng.exponential(mean_val / 2.0) + rng.exponential(mean_val / 2.0)
    
    def arrive_at_station(now, jtype, tidx):
        station = ROUTINGS[jtype][tidx]
        if busy[station] < machines_per_station[station]:
            # Langsung dilayani (delay = 0)
            if now >= WARMUP_HOURS:
                count_tasks_by_type[jtype] += 1
            busy[station] += 1
            stime = gen_erlang2(SERVICE_MEANS[jtype][tidx])
            schedule_event(now + stime, 2, station, jtype, tidx)
        else:
            # Masuk antrean FIFO
            queues[station].append((now, jtype, tidx))
            
    # Jadwalkan kedatangan job pertama dan akhir simulasi
    first_arrival = rng.exponential(MEAN_INTERARRIVAL)
    schedule_event(first_arrival, 1)
    schedule_event(TOTAL_HOURS, 3)
    
    while event_list:
        now, _, etype, station, jtype, tidx = heapq.heappop(event_list)
        
        if etype == 3:
            # Akhir simulasi
            break
            
        elif etype == 1:
            # Kedatangan job baru ke sistem
            next_arr = now + rng.exponential(MEAN_INTERARRIVAL)
            schedule_event(next_arr, 1)
            
            new_jtype = rng.choice([0, 1, 2], p=JOB_PROBS)
            arrive_at_station(now, new_jtype, 0)
            
        elif etype == 2:
            # Selesai diproses di stasiun saat ini
            if queues[station]:
                arr_q_time, q_jtype, q_tidx = queues[station].pop(0)
                if now >= WARMUP_HOURS:
                    delay = now - arr_q_time
                    total_delay_by_type[q_jtype] += delay
                    count_tasks_by_type[q_jtype] += 1
                stime = gen_erlang2(SERVICE_MEANS[q_jtype][q_tidx])
                schedule_event(now + stime, 2, station, q_jtype, q_tidx)
            else:
                busy[station] -= 1
                
            # Pindah ke task berikutnya jika belum selesai seluruh routing
            if tidx + 1 < len(ROUTINGS[jtype]):
                arrive_at_station(now, jtype, tidx + 1)
                
    # Hitung expected overall average job total delay (mengikuti metode Sec. 2.7.2)
    overall_avg_delay = 0.0
    for jt in range(3):
        num_tasks = len(ROUTINGS[jt])
        avg_delay_per_task = total_delay_by_type[jt] / count_tasks_by_type[jt]
        avg_total_delay_jt = avg_delay_per_task * num_tasks
        overall_avg_delay += JOB_PROBS[jt] * avg_total_delay_jt
        
    return overall_avg_delay

# ==========================================
# 3. PROSEDUR DUDEWICZ-DALAL (SEC. 10.4.1)
# ==========================================
def run_dd_procedure():
    alternatives = [
        ("Sistem 1 (+1 mesin di Station 1)", [4, 2, 4, 3, 1]),
        ("Sistem 2 (+1 mesin di Station 2)", [3, 3, 4, 3, 1]),
        ("Sistem 3 (+1 mesin di Station 4)", [3, 2, 4, 4, 1]),
    ]
    
    n0 = 20
    d_star = 1.0
    h1 = 2.342  # Table 10.11 untuk k=3, P*=0.90, n0=20
    
    print("=== TAHAP 1: INITIAL SAMPLING (n0 = 20) ===")
    stage1_data = []
    X1_bar = []
    S2 = []
    N_total = []
    
    seed_counter = 1000
    for i, (name, config) in enumerate(alternatives):
        obs = []
        for _ in range(n0):
            # Independen antar replikasi dan antar sistem
            obs.append(run_jobshop_replication(config, seed_counter))
            seed_counter += 1
            
        mean_1 = np.mean(obs)
        var_1 = np.var(obs, ddof=1)
        n_req = max(n0 + 1, math.ceil((h1**2 * var_1) / (d_star**2)))
        
        stage1_data.append(obs)
        X1_bar.append(mean_1)
        S2.append(var_1)
        N_total.append(n_req)
        
        print(f"{name}:")
        print(f"  X_bar^(1)(20) = {mean_1:.4f} jam | S_i^2(20) = {var_1:.4f} | N_i = {n_req}")
        
    print("\n=== TAHAP 2: SECOND-STAGE SAMPLING & WEIGHTED MEAN ===")
    results = []
    for i, (name, config) in enumerate(alternatives):
        n_extra = N_total[i] - n0
        obs2 = []
        for _ in range(n_extra):
            obs2.append(run_jobshop_replication(config, seed_counter))
            seed_counter += 1
            
        mean_2 = np.mean(obs2)
        
        # Hitung bobot W_i1 dan W_i2 sesuai Persamaan di halaman 572
        term_inside = 1.0 - ((N_total[i] - n0) * (d_star**2)) / (h1**2 * S2[i])
        # Pastikan tidak negatif akibat floating-point precision
        term_inside = max(0.0, term_inside)
        W_i1 = (n0 / N_total[i]) * (1.0 + math.sqrt(1.0 - (N_total[i] / n0) * term_inside))
        W_i2 = 1.0 - W_i1
        
        X_tilde = W_i1 * X1_bar[i] + W_i2 * mean_2
        results.append((X_tilde, name, X1_bar[i], S2[i], N_total[i], mean_2, W_i1, W_i2))
        
        print(f"{name}:")
        print(f"  Tambahan replikasi = {n_extra} | X_bar^(2) = {mean_2:.4f}")
        print(f"  W_i1 = {W_i1:.4f}, W_i2 = {W_i2:.4f} | Weighted Mean X_tilde = {X_tilde:.4f} jam")
        
    best = min(results, key=lambda x: x[0])
    print("\n=== KEPUTUSAN AKHIR ===")
    print(f"Rekomendasi terbaik (delay terkecil): {best[1]} dengan X_tilde = {best[0]:.4f} jam")

if __name__ == "__main__":
    run_dd_procedure()