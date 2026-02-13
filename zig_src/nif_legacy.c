/**
 * nif_legacy.c - Legacy list-based CPU NIFs (backward compatibility)
 * SIMD dot/sum/scale/add/mul, list-based matmul, backend info, CPU topology.
 */

#include "viva_nif.h"

ERL_NIF_TERM nif_simd_dot(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  if (argc != 2)
    return enif_make_badarg(env);
  unsigned la, lb;
  double *a = list_to_doubles(env, argv[0], &la);
  if (!a)
    return make_error(env, "invalid_input");
  double *b = list_to_doubles(env, argv[1], &lb);
  if (!b) {
    free(a);
    return make_error(env, "invalid_input");
  }
  if (la != lb) {
    free(a);
    free(b);
    return make_error(env, "length_mismatch");
  }
  double r = vt_simd_dot(a, b, la);
  free(a);
  free(b);
  return make_ok(env, enif_make_double(env, r));
}

ERL_NIF_TERM nif_simd_sum(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  if (argc != 1)
    return enif_make_badarg(env);
  unsigned len;
  double *d = list_to_doubles(env, argv[0], &len);
  if (!d)
    return make_error(env, "invalid_input");
  double r = vt_simd_sum(d, len);
  free(d);
  return make_ok(env, enif_make_double(env, r));
}

ERL_NIF_TERM nif_simd_scale(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  if (argc != 2)
    return enif_make_badarg(env);
  unsigned len;
  double *d = list_to_doubles(env, argv[0], &len);
  if (!d)
    return make_error(env, "invalid_input");
  int ok;
  double s = get_number(env, argv[1], &ok);
  if (!ok) {
    free(d);
    return make_error(env, "invalid_scalar");
  }
  double *r = (double *)malloc(len * sizeof(double));
  if (!r) {
    free(d);
    return make_error(env, "out_of_memory");
  }
  vt_simd_scale(d, s, r, len);
  ERL_NIF_TERM rl = doubles_to_list(env, r, len);
  free(d);
  free(r);
  return make_ok(env, rl);
}

ERL_NIF_TERM nif_simd_add(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  if (argc != 2)
    return enif_make_badarg(env);
  unsigned la, lb;
  double *a = list_to_doubles(env, argv[0], &la);
  if (!a)
    return make_error(env, "invalid_input");
  double *b = list_to_doubles(env, argv[1], &lb);
  if (!b) {
    free(a);
    return make_error(env, "invalid_input");
  }
  if (la != lb) {
    free(a);
    free(b);
    return make_error(env, "length_mismatch");
  }
  double *r = (double *)malloc(la * sizeof(double));
  if (!r) {
    free(a);
    free(b);
    return make_error(env, "out_of_memory");
  }
  vt_simd_add(a, b, r, la);
  ERL_NIF_TERM rl = doubles_to_list(env, r, la);
  free(a);
  free(b);
  free(r);
  return make_ok(env, rl);
}

ERL_NIF_TERM nif_simd_mul(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  if (argc != 2)
    return enif_make_badarg(env);
  unsigned la, lb;
  double *a = list_to_doubles(env, argv[0], &la);
  if (!a)
    return make_error(env, "invalid_input");
  double *b = list_to_doubles(env, argv[1], &lb);
  if (!b) {
    free(a);
    return make_error(env, "invalid_input");
  }
  if (la != lb) {
    free(a);
    free(b);
    return make_error(env, "length_mismatch");
  }
  double *r = (double *)malloc(la * sizeof(double));
  if (!r) {
    free(a);
    free(b);
    return make_error(env, "out_of_memory");
  }
  vt_simd_mul(a, b, r, la);
  ERL_NIF_TERM rl = doubles_to_list(env, r, la);
  free(a);
  free(b);
  free(r);
  return make_ok(env, rl);
}

ERL_NIF_TERM nif_simd_matmul(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  if (argc != 5)
    return enif_make_badarg(env);
  int mi, ni, ki;
  if (!enif_get_int(env, argv[2], &mi) || !enif_get_int(env, argv[3], &ni) ||
      !enif_get_int(env, argv[4], &ki))
    return make_error(env, "invalid_dimensions");
  unsigned la, lb;
  double *a = list_to_doubles(env, argv[0], &la);
  if (!a)
    return make_error(env, "invalid_input");
  double *b = list_to_doubles(env, argv[1], &lb);
  if (!b) {
    free(a);
    return make_error(env, "invalid_input");
  }
  if ((int)la != mi * ki || (int)lb != ki * ni) {
    free(a);
    free(b);
    return make_error(env, "size_mismatch");
  }
  double *c = (double *)malloc(mi * ni * sizeof(double));
  if (!c) {
    free(a);
    free(b);
    return make_error(env, "out_of_memory");
  }
#if defined(_WIN32) || defined(USE_MKL_DIRECT)
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              mi, ni, ki, 1.0, a, ki, b, ni, 0.0, c, ni);
#else
  if (g_dgemm) {
    blas_dgemm(mi, ni, ki, 1.0, a, ki, b, ni, 0.0, c, ni);
  } else {
    free(a); free(b); free(c);
    return make_error(env, "no_blas_backend");
  }
#endif
  ERL_NIF_TERM rl = doubles_to_list(env, c, mi * ni);
  free(a);
  free(b);
  free(c);
  return make_ok(env, rl);
}

ERL_NIF_TERM nif_simd_available(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  (void)argv;
  return enif_make_atom(env, "true");
}

ERL_NIF_TERM nif_backend_info(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  (void)argc;
  (void)argv;
  char info[512];
#if defined(_WIN32) || defined(USE_MKL_DIRECT)
  const char *blas_name = "Intel MKL";
#else
  const char *blas_name = g_blas_name ? g_blas_name : "Zig GEMM";
#endif
  snprintf(info, sizeof(info),
           "%s + Zig SIMD (Vec8 f64) | %d cores (%d logical) | L2:%dKB L3:%dKB | %s%s| %d threads",
           blas_name,
           g_cpu_info.physical_cores,
           g_cpu_info.logical_cpus,
           g_cpu_info.l2_cache_kb,
           g_cpu_info.l3_cache_kb,
           g_cpu_info.has_avx512 ? "AVX-512 " : (g_cpu_info.has_avx2 ? "AVX2 " : ""),
           g_cpu_info.has_hybrid ? "Hybrid " : "",
           g_cpu_info.optimal_threads);
  ErlNifBinary bin;
  if (!enif_alloc_binary(strlen(info), &bin))
    return enif_make_atom(env, "error");
  memcpy(bin.data, info, strlen(info));
  return enif_make_binary(env, &bin);
}

/** cpu_topology() -> {ok, Map} */
ERL_NIF_TERM nif_cpu_topology(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  (void)argc;
  (void)argv;

  ERL_NIF_TERM keys[12], vals[12];
  int i = 0;

  keys[i] = enif_make_atom(env, "logical_cpus");
  vals[i++] = enif_make_int(env, g_cpu_info.logical_cpus);

  keys[i] = enif_make_atom(env, "physical_cores");
  vals[i++] = enif_make_int(env, g_cpu_info.physical_cores);

  keys[i] = enif_make_atom(env, "sockets");
  vals[i++] = enif_make_int(env, g_cpu_info.sockets);

  keys[i] = enif_make_atom(env, "threads_per_core");
  vals[i++] = enif_make_int(env, g_cpu_info.threads_per_core);

  keys[i] = enif_make_atom(env, "l1_cache_kb");
  vals[i++] = enif_make_int(env, g_cpu_info.l1_cache_kb);

  keys[i] = enif_make_atom(env, "l2_cache_kb");
  vals[i++] = enif_make_int(env, g_cpu_info.l2_cache_kb);

  keys[i] = enif_make_atom(env, "l3_cache_kb");
  vals[i++] = enif_make_int(env, g_cpu_info.l3_cache_kb);

  keys[i] = enif_make_atom(env, "has_avx2");
  vals[i++] = enif_make_atom(env, g_cpu_info.has_avx2 ? "true" : "false");

  keys[i] = enif_make_atom(env, "has_avx512");
  vals[i++] = enif_make_atom(env, g_cpu_info.has_avx512 ? "true" : "false");

  keys[i] = enif_make_atom(env, "has_hybrid");
  vals[i++] = enif_make_atom(env, g_cpu_info.has_hybrid ? "true" : "false");

  keys[i] = enif_make_atom(env, "optimal_threads");
  vals[i++] = enif_make_int(env, g_cpu_info.optimal_threads);

  if (g_cpu_info.has_hybrid) {
    keys[i] = enif_make_atom(env, "p_cores");
    vals[i++] = enif_make_int(env, g_cpu_info.p_cores);
  }

  ERL_NIF_TERM map;
  enif_make_map_from_arrays(env, keys, vals, i, &map);
  return make_ok(env, map);
}
