/**
 * nif_platform.c - BLAS backend detection and CPU topology
 *
 * Extracted from nif_entry.c. Contains:
 *   - Dynamic BLAS backend selection (MKL > OpenBLAS > Zig GEMM)
 *   - CPU topology detection (Linux sysfs + Windows)
 *   - Thread affinity helpers
 *   - vt_* export functions for Zig
 */

#include "viva_nif.h"

/* =========================================================================
 * BLAS Backend (Linux: dynamic loading, Windows: direct MKL)
 * ========================================================================= */

#ifndef _WIN32
#ifndef USE_MKL_DIRECT

/* Global BLAS state (extern in viva_nif.h) */
BlasBackend g_blas_backend = BLAS_ZIG_GEMM;
void *g_blas_handle = NULL;
dgemm_fn g_dgemm = NULL;
set_threads_fn g_set_threads = NULL;
const char *g_blas_name = "Zig GEMM";
int g_blas_detected = 0;

/* Try to load a BLAS library dynamically */
static int try_load_blas(const char *libname, const char *backend_name, BlasBackend backend_type) {
  void *handle = dlopen(libname, RTLD_NOW | RTLD_LOCAL);
  if (!handle) return 0;

  dgemm_fn dgemm = (dgemm_fn)dlsym(handle, "cblas_dgemm");
  if (!dgemm) {
    dlclose(handle);
    return 0;
  }

  g_blas_handle = handle;
  g_dgemm = dgemm;
  g_blas_backend = backend_type;
  g_blas_name = backend_name;

  if (backend_type == BLAS_MKL) {
    g_set_threads = (set_threads_fn)dlsym(handle, "mkl_set_num_threads");
  } else {
    g_set_threads = (set_threads_fn)dlsym(handle, "openblas_set_num_threads");
  }

  if (g_set_threads) {
    int ncpus = sysconf(_SC_NPROCESSORS_ONLN);
    int optimal = ncpus > 0 ? ncpus : 16;
    g_set_threads(optimal);
    fprintf(stderr, "[viva_tensor] BLAS threads: %d\n", optimal);
  }

  return 1;
}

/* Detect and load the best available BLAS backend */
void detect_blas_backend(void) {
  if (g_blas_detected) return;
  g_blas_detected = 1;

  /* Priority: Intel MKL > OpenBLAS-tuned > OpenBLAS system > Zig GEMM */

  /* 1. Tuned OpenBLAS (HASWELL-optimized) */
  char tuned_path[512];
  if (getcwd(tuned_path, sizeof(tuned_path) - 100)) {
    strcat(tuned_path, "/deps/openblas-tuned/lib/libopenblas.so");
  } else {
    strcpy(tuned_path, "deps/openblas-tuned/lib/libopenblas.so");
  }
  if (try_load_blas(tuned_path, "OpenBLAS-HASWELL", BLAS_OPENBLAS_TUNED)) {
    fprintf(stderr, "[viva_tensor] Backend: OpenBLAS-HASWELL (tuned, 500+ GFLOPS)\n");
    return;
  }

  /* 3. System OpenBLAS */
  if (try_load_blas("libopenblas.so.0", "OpenBLAS", BLAS_OPENBLAS)) {
    fprintf(stderr, "[viva_tensor] Backend: OpenBLAS system\n");
    return;
  }
  if (try_load_blas("libopenblas.so", "OpenBLAS", BLAS_OPENBLAS)) {
    fprintf(stderr, "[viva_tensor] Backend: OpenBLAS system\n");
    return;
  }

  /* 4. Fallback to Zig GEMM */
  fprintf(stderr, "[viva_tensor] Backend: Zig GEMM (native, 200+ GFLOPS)\n");
}

void blas_dgemm(int M, int N, int K, double alpha,
                const double *A, int lda,
                const double *B, int ldb,
                double beta, double *C, int ldc) {
  if (g_dgemm) {
    g_dgemm(101, 111, 111, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
}

void blas_set_threads(int n) {
  if (g_set_threads) {
    g_set_threads(n);
  }
}

#endif /* !USE_MKL_DIRECT */
#endif /* !_WIN32 */

/* =========================================================================
 * CPU Topology Detection
 * ========================================================================= */

/* Global CPU topology (extern in viva_nif.h) */
CpuTopology g_cpu_info = {0};
int g_cpu_detected = 0;

#ifdef _WIN32
static void detect_cpu_topology_windows(void) {
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  g_cpu_info.logical_cpus = sysInfo.dwNumberOfProcessors;

  DWORD bufLen = 0;
  GetLogicalProcessorInformation(NULL, &bufLen);
  if (bufLen == 0) {
    g_cpu_info.physical_cores = g_cpu_info.logical_cpus / 2;
    g_cpu_info.threads_per_core = 2;
    goto compute_optimal;
  }

  SYSTEM_LOGICAL_PROCESSOR_INFORMATION *buf =
    (SYSTEM_LOGICAL_PROCESSOR_INFORMATION *)malloc(bufLen);
  if (!buf) {
    g_cpu_info.physical_cores = g_cpu_info.logical_cpus / 2;
    g_cpu_info.threads_per_core = 2;
    goto compute_optimal;
  }

  if (GetLogicalProcessorInformation(buf, &bufLen)) {
    int cores = 0, l1 = 0, l2 = 0, l3 = 0;
    DWORD offset = 0;
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION *ptr = buf;
    while (offset < bufLen) {
      switch (ptr->Relationship) {
        case RelationProcessorCore: cores++; break;
        case RelationCache:
          if (ptr->Cache.Level == 1 && ptr->Cache.Type == CacheData)
            l1 = ptr->Cache.Size / 1024;
          else if (ptr->Cache.Level == 2)
            l2 = ptr->Cache.Size / 1024;
          else if (ptr->Cache.Level == 3)
            l3 = ptr->Cache.Size / 1024;
          break;
        default: break;
      }
      offset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
      ptr++;
    }
    g_cpu_info.physical_cores = cores > 0 ? cores : g_cpu_info.logical_cpus / 2;
    g_cpu_info.l1_cache_kb = l1;
    g_cpu_info.l2_cache_kb = l2;
    g_cpu_info.l3_cache_kb = l3;
  }
  free(buf);

  g_cpu_info.threads_per_core = g_cpu_info.logical_cpus / g_cpu_info.physical_cores;
  g_cpu_info.sockets = 1;

  int cpuInfo[4];
  __cpuid(cpuInfo, 7);
  g_cpu_info.has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;
  g_cpu_info.has_avx512 = (cpuInfo[1] & (1 << 16)) != 0;

  __cpuid(cpuInfo, 0x1A);
  g_cpu_info.has_hybrid = (cpuInfo[0] != 0);
  if (g_cpu_info.has_hybrid) {
    g_cpu_info.p_cores = (g_cpu_info.physical_cores * 2) / 3;
    g_cpu_info.e_cores = g_cpu_info.physical_cores - g_cpu_info.p_cores;
  }

compute_optimal:
  if (g_cpu_info.has_hybrid) {
    g_cpu_info.optimal_threads = g_cpu_info.logical_cpus;
  } else {
    g_cpu_info.optimal_threads = g_cpu_info.logical_cpus;
  }
}
#else /* Linux/macOS */
static void detect_cpu_topology_linux(void) {
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (!f) {
    g_cpu_info.logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    g_cpu_info.physical_cores = g_cpu_info.logical_cpus / 2;
    g_cpu_info.threads_per_core = 2;
    goto compute_optimal;
  }

  char line[256];
  int logical = 0, phys_ids[64] = {0}, core_ids[64] = {0};
  int unique_phys = 0, unique_cores = 0;
  int l1 = 0, l2 = 0, l3 = 0;
  int avx2 = 0, avx512 = 0;

  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "processor", 9) == 0) {
      logical++;
    } else if (strncmp(line, "physical id", 11) == 0) {
      int id;
      if (sscanf(line, "physical id : %d", &id) == 1 && id < 64) {
        if (!phys_ids[id]) { phys_ids[id] = 1; unique_phys++; }
      }
    } else if (strncmp(line, "core id", 7) == 0) {
      int id;
      if (sscanf(line, "core id : %d", &id) == 1 && id < 64) {
        if (!core_ids[id]) { core_ids[id] = 1; unique_cores++; }
      }
    } else if (strncmp(line, "cache size", 10) == 0) {
      int sz;
      if (sscanf(line, "cache size : %d KB", &sz) == 1) l3 = sz;
    } else if (strstr(line, "avx2")) {
      avx2 = 1;
    } else if (strstr(line, "avx512")) {
      avx512 = 1;
    }
  }
  fclose(f);

  g_cpu_info.logical_cpus = logical > 0 ? logical : sysconf(_SC_NPROCESSORS_ONLN);
  g_cpu_info.sockets = unique_phys > 0 ? unique_phys : 1;
  g_cpu_info.physical_cores = unique_cores > 0 ? unique_cores * g_cpu_info.sockets
                                               : g_cpu_info.logical_cpus / 2;
  g_cpu_info.threads_per_core = g_cpu_info.logical_cpus / g_cpu_info.physical_cores;
  g_cpu_info.l3_cache_kb = l3;
  g_cpu_info.has_avx2 = avx2;
  g_cpu_info.has_avx512 = avx512;

  /* Read cache info from sysfs */
  FILE *cache_f;
  char buf[64];

  cache_f = fopen("/sys/devices/system/cpu/cpu0/cache/index0/size", "r");
  if (cache_f) {
    if (fgets(buf, sizeof(buf), cache_f)) {
      sscanf(buf, "%dK", &l1);
      g_cpu_info.l1_cache_kb = l1;
    }
    fclose(cache_f);
  }

  cache_f = fopen("/sys/devices/system/cpu/cpu0/cache/index2/size", "r");
  if (cache_f) {
    if (fgets(buf, sizeof(buf), cache_f)) {
      sscanf(buf, "%dK", &l2);
      g_cpu_info.l2_cache_kb = l2;
    }
    fclose(cache_f);
  }

  /* Check hybrid architecture (Linux 5.8+) */
  cache_f = fopen("/sys/devices/system/cpu/cpu0/cpu_capacity", "r");
  if (cache_f) {
    int cap;
    if (fgets(buf, sizeof(buf), cache_f) && sscanf(buf, "%d", &cap) == 1) {
      if (cap < 1024) g_cpu_info.has_hybrid = 1;
    }
    fclose(cache_f);
  }

  if (g_cpu_info.has_hybrid) {
    int p_count = 0, e_count = 0;
    for (int i = 0; i < g_cpu_info.logical_cpus; i++) {
      char path[128];
      snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpu_capacity", i);
      cache_f = fopen(path, "r");
      if (cache_f) {
        int cap;
        if (fgets(buf, sizeof(buf), cache_f) && sscanf(buf, "%d", &cap) == 1) {
          if (cap >= 1024) p_count++;
          else e_count++;
        }
        fclose(cache_f);
      }
    }
    g_cpu_info.p_cores = p_count / g_cpu_info.threads_per_core;
    g_cpu_info.e_cores = e_count;
  }

compute_optimal:
  if (g_cpu_info.has_hybrid && g_cpu_info.p_cores > 0) {
    g_cpu_info.optimal_threads = g_cpu_info.p_cores * g_cpu_info.threads_per_core;
  } else {
    g_cpu_info.optimal_threads = g_cpu_info.logical_cpus;
  }
}
#endif

void detect_cpu_topology(void) {
  if (g_cpu_detected) return;
#ifdef _WIN32
  detect_cpu_topology_windows();
#else
  detect_cpu_topology_linux();
#endif
  g_cpu_detected = 1;
}

/* =========================================================================
 * Zig Export Functions
 * ========================================================================= */

int vt_get_optimal_threads(void) {
  return g_cpu_info.optimal_threads > 0 ? g_cpu_info.optimal_threads : 8;
}

int vt_get_physical_cores(void) {
  return g_cpu_info.physical_cores > 0 ? g_cpu_info.physical_cores : 4;
}

int vt_get_logical_cpus(void) {
  return g_cpu_info.logical_cpus > 0 ? g_cpu_info.logical_cpus : 8;
}

int vt_get_l2_cache_kb(void) {
  return g_cpu_info.l2_cache_kb > 0 ? g_cpu_info.l2_cache_kb : 256;
}

int vt_get_l3_cache_kb(void) {
  return g_cpu_info.l3_cache_kb > 0 ? g_cpu_info.l3_cache_kb : 8192;
}

int vt_is_hybrid_cpu(void) {
  return g_cpu_info.has_hybrid;
}

int vt_has_avx512(void) {
  return g_cpu_info.has_avx512;
}

/* =========================================================================
 * Thread Affinity Helpers (called from Zig)
 * ========================================================================= */

#ifdef _WIN32
int vt_set_thread_affinity(void* thread_handle, int core_id) {
  DWORD_PTR mask = 1ULL << core_id;
  return SetThreadAffinityMask((HANDLE)thread_handle, mask) != 0;
}
#else
int vt_set_thread_affinity_self(int core_id) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);
  return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
}
#endif
