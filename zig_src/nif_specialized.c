/* =========================================================================
 * nif_specialized.c - Specialized NIF backends
 *
 * Extracted from nif_entry.c. Contains:
 *   - Resonance Kernels (Log-Number System multiply/power)
 *   - LNS Tensor (True Log-Number System with f32 IADD trick)
 *   - Horde Physics Engine (SoA entity system)
 *   - HDC (Hyperdimensional Computing with binary vectors)
 *
 * All resource types (LNS_RESOURCE, HORDE_RESOURCE, HDC_RESOURCE) are
 * declared extern and initialized in nif_entry.c's nif_load callback.
 * ========================================================================= */

#include "viva_nif.h"

/* =========================================================================
 * Resonance Kernels (Log-Number System)
 * "Multiplicação como soma no domínio logarítmico"
 * ========================================================================= */

/** nt_resonance_mul(RefA, RefB) -> {ok, RefC}
 * LNS element-wise multiply: result = sign * exp(log|a| + log|b|)
 * Turns multiply into add in log domain. Better precision for chains. */
ERL_NIF_TERM nt_resonance_mul(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  NativeTensor *b = get_tensor(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_tensor");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_resonance_mul(a->data, b->data, c->data, (size_t)a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/** nt_resonance_power(Ref, Exponent) -> {ok, RefC}
 * LNS power: result = sign * |x|^exponent via exp(exponent * log|x|)
 * Power = multiply in log domain. Sign preserved for bipolar states. */
ERL_NIF_TERM nt_resonance_power(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *a = get_tensor(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_tensor");

  int ok;
  double exponent = get_number(env, argv[1], &ok);
  if (!ok)
    return make_error(env, "invalid_exponent");

  NativeTensor *c = alloc_tensor_uninit(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_resonance_power(a->data, exponent, c->data, (size_t)a->size);
  return make_ok(env, make_tensor_term(env, c));
}

/* =========================================================================
 * LNS Tensor (True Log-Number System) - f32 via IADD
 * 8x throughput vs FMA by turning multiply into integer add
 * ========================================================================= */

/* LnsTensor struct defined in viva_nif.h */

ErlNifResourceType *LNS_RESOURCE = NULL;

void lns_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  LnsTensor *t = (LnsTensor *)obj;
  if (t->data)
    aligned_tensor_free(t->data);
  if (t->shape)
    free(t->shape);
}

LnsTensor *alloc_lns(int ndim, const int *shape) {
  LnsTensor *t =
      (LnsTensor *)enif_alloc_resource(LNS_RESOURCE, sizeof(LnsTensor));
  if (!t)
    return NULL;
  t->ndim = ndim;
  t->size = 1;
  for (int i = 0; i < ndim; i++)
    t->size *= shape[i];
  t->shape = (int *)malloc(ndim * sizeof(int));
  if (!t->shape) {
    enif_release_resource(t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));
  t->data = (float *)aligned_tensor_alloc(t->size * sizeof(float));
  if (!t->data) {
    free(t->shape);
    enif_release_resource(t);
    return NULL;
  }
  return t;
}

LnsTensor *get_lns(ErlNifEnv *env, ERL_NIF_TERM term) {
  LnsTensor *t;
  if (!enif_get_resource(env, term, LNS_RESOURCE, (void **)&t))
    return NULL;
  return t;
}

/** lns_from_f64(NativeTensorRef) -> {ok, LnsRef}
 * Convert f64 tensor to f32 LNS tensor */
ERL_NIF_TERM lns_from_f64(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  NativeTensor *src = get_tensor(env, argv[0]);
  if (!src)
    return make_error(env, "invalid_tensor");

  LnsTensor *dst = alloc_lns(src->ndim, src->shape);
  if (!dst)
    return make_error(env, "out_of_memory");

  for (int i = 0; i < src->size; i++)
    dst->data[i] = (float)src->data[i];

  ERL_NIF_TERM term = enif_make_resource(env, dst);
  enif_release_resource(dst);
  return make_ok(env, term);
}

/** lns_to_f64(LnsRef) -> {ok, NativeTensorRef}
 * Convert back to f64 */
ERL_NIF_TERM lns_to_f64(ErlNifEnv *env, int argc,
                               const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *src = get_lns(env, argv[0]);
  if (!src)
    return make_error(env, "invalid_lns");

  NativeTensor *dst = alloc_tensor_uninit(src->ndim, src->shape);
  if (!dst)
    return make_error(env, "out_of_memory");

  for (int i = 0; i < src->size; i++)
    dst->data[i] = (double)src->data[i];

  return make_ok(env, make_tensor_term(env, dst));
}

/** lns_mul(LnsA, LnsB) -> {ok, LnsC}
 * Fast LNS multiply via IADD (~11% max error) */
ERL_NIF_TERM lns_mul(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *a = get_lns(env, argv[0]);
  LnsTensor *b = get_lns(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_lns");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  LnsTensor *c = alloc_lns(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_lns_mul_f32(a->data, b->data, c->data, (size_t)a->size);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** lns_mul_corrected(LnsA, LnsB) -> {ok, LnsC}
 * Mitchell's corrected LNS multiply (~2% max error) */
ERL_NIF_TERM lns_mul_corrected(ErlNifEnv *env, int argc,
                                      const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *a = get_lns(env, argv[0]);
  LnsTensor *b = get_lns(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_lns");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  LnsTensor *c = alloc_lns(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_lns_mul_corrected_f32(a->data, b->data, c->data, (size_t)a->size);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** lns_div(LnsA, LnsB) -> {ok, LnsC}
 * LNS division via ISUB */
ERL_NIF_TERM lns_div(ErlNifEnv *env, int argc,
                            const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *a = get_lns(env, argv[0]);
  LnsTensor *b = get_lns(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_lns");
  if (a->size != b->size)
    return make_error(env, "size_mismatch");

  LnsTensor *c = alloc_lns(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_lns_div_f32(a->data, b->data, c->data, (size_t)a->size);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** lns_sqrt(Lns) -> {ok, LnsC}
 * LNS sqrt via bit shift */
ERL_NIF_TERM lns_sqrt(ErlNifEnv *env, int argc,
                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *a = get_lns(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_lns");

  LnsTensor *c = alloc_lns(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_lns_sqrt_f32(a->data, c->data, (size_t)a->size);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** lns_rsqrt(Lns) -> {ok, LnsC}
 * Fast inverse sqrt (Quake III trick) */
ERL_NIF_TERM lns_rsqrt(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
  (void)argc;
  LnsTensor *a = get_lns(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_lns");

  LnsTensor *c = alloc_lns(a->ndim, a->shape);
  if (!c)
    return make_error(env, "out_of_memory");

  vt_lns_rsqrt_f32(a->data, c->data, (size_t)a->size);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/* =========================================================================
 * Horde - SoA Physics Engine
 * 10K+ entities at 60fps with zero GC pressure
 * ========================================================================= */

/* Horde struct defined in viva_nif.h */

ErlNifResourceType *HORDE_RESOURCE = NULL;

void horde_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  Horde *h = (Horde *)obj;
  if (h->positions)
    aligned_tensor_free(h->positions);
  if (h->velocities)
    aligned_tensor_free(h->velocities);
  if (h->accelerations)
    aligned_tensor_free(h->accelerations);
}

/** horde_create(EntityCount, Dims) -> {ok, HordeRef} */
ERL_NIF_TERM horde_create(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  int count, dims;
  if (!enif_get_int(env, argv[0], &count) ||
      !enif_get_int(env, argv[1], &dims) || count <= 0 || dims < 1 || dims > 3)
    return make_error(env, "invalid_params");

  Horde *h = (Horde *)enif_alloc_resource(HORDE_RESOURCE, sizeof(Horde));
  if (!h)
    return make_error(env, "out_of_memory");

  h->entity_count = count;
  h->dims = dims;
  size_t size = count * dims * sizeof(double);

  h->positions = (double *)aligned_tensor_alloc(size);
  h->velocities = (double *)aligned_tensor_alloc(size);
  h->accelerations = NULL; /* Lazy alloc */

  if (!h->positions || !h->velocities) {
    if (h->positions)
      aligned_tensor_free(h->positions);
    if (h->velocities)
      aligned_tensor_free(h->velocities);
    enif_release_resource(h);
    return make_error(env, "out_of_memory");
  }

  memset(h->positions, 0, size);
  memset(h->velocities, 0, size);

  ERL_NIF_TERM term = enif_make_resource(env, h);
  enif_release_resource(h);
  return make_ok(env, term);
}

Horde *get_horde(ErlNifEnv *env, ERL_NIF_TERM term) {
  Horde *h;
  if (!enif_get_resource(env, term, HORDE_RESOURCE, (void **)&h))
    return NULL;
  return h;
}

/** horde_set_positions(HordeRef, DataList) -> ok */
ERL_NIF_TERM horde_set_positions(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  unsigned len;
  double *data = list_to_doubles(env, argv[1], &len);
  if (!data)
    return make_error(env, "invalid_data");

  size_t expected = h->entity_count * h->dims;
  if (len != expected) {
    free(data);
    return make_error(env, "size_mismatch");
  }

  memcpy(h->positions, data, len * sizeof(double));
  free(data);
  return make_ok_nil(env);
}

/** horde_set_velocities(HordeRef, DataList) -> ok */
ERL_NIF_TERM horde_set_velocities(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  unsigned len;
  double *data = list_to_doubles(env, argv[1], &len);
  if (!data)
    return make_error(env, "invalid_data");

  size_t expected = h->entity_count * h->dims;
  if (len != expected) {
    free(data);
    return make_error(env, "size_mismatch");
  }

  memcpy(h->velocities, data, len * sizeof(double));
  free(data);
  return make_ok_nil(env);
}

/** horde_integrate(HordeRef, Dt) -> ok
 * Euler step: pos += vel * dt (FMA) */
ERL_NIF_TERM horde_integrate_nif(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  double dt;
  if (!enif_get_double(env, argv[1], &dt))
    return make_error(env, "invalid_dt");

  vt_horde_integrate(h->positions, h->velocities, dt,
                     h->entity_count * h->dims);
  return make_ok_nil(env);
}

/** horde_dampen(HordeRef, Friction) -> ok */
ERL_NIF_TERM horde_dampen_nif(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  double friction;
  if (!enif_get_double(env, argv[1], &friction))
    return make_error(env, "invalid_friction");

  vt_horde_dampen(h->velocities, friction, h->entity_count * h->dims);
  return make_ok_nil(env);
}

/** horde_wrap(HordeRef, MaxBound) -> ok
 * Toroidal boundary conditions */
ERL_NIF_TERM horde_wrap_nif(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  double max_bound;
  if (!enif_get_double(env, argv[1], &max_bound))
    return make_error(env, "invalid_bound");

  vt_horde_wrap(h->positions, max_bound, h->entity_count * h->dims);
  return make_ok_nil(env);
}

/** horde_get_positions(HordeRef) -> {ok, List} */
ERL_NIF_TERM horde_get_positions(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  return make_ok(env,
                 doubles_to_list(env, h->positions, h->entity_count * h->dims));
}

/** horde_get_velocities(HordeRef) -> {ok, List} */
ERL_NIF_TERM horde_get_velocities(ErlNifEnv *env, int argc,
                                         const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  return make_ok(
      env, doubles_to_list(env, h->velocities, h->entity_count * h->dims));
}

/** horde_count(HordeRef) -> {ok, Int} */
ERL_NIF_TERM horde_count_nif(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  return make_ok(env, enif_make_int(env, h->entity_count));
}

/** horde_kinetic_energy(HordeRef) -> {ok, Float} */
ERL_NIF_TERM horde_kinetic_energy_nif(ErlNifEnv *env, int argc,
                                             const ERL_NIF_TERM argv[]) {
  (void)argc;
  Horde *h = get_horde(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_horde");

  double ke = vt_horde_kinetic_energy(h->velocities, h->entity_count * h->dims);
  return make_ok(env, enif_make_double(env, ke));
}

/* =========================================================================
 * HDC - Hyperdimensional Computing
 * One-shot learning via binary vectors and popcount similarity
 * ========================================================================= */

/* HdcVector struct defined in viva_nif.h */

ErlNifResourceType *HDC_RESOURCE = NULL;

void hdc_destructor(ErlNifEnv *env, void *obj) {
  (void)env;
  HdcVector *h = (HdcVector *)obj;
  if (h->data)
    aligned_tensor_free(h->data);
}

HdcVector *get_hdc(ErlNifEnv *env, ERL_NIF_TERM term) {
  HdcVector *h;
  if (!enif_get_resource(env, term, HDC_RESOURCE, (void **)&h))
    return NULL;
  return h;
}

/** hdc_create(Dim) -> {ok, HdcRef}
 * Dim must be multiple of 64 */
ERL_NIF_TERM hdc_create_nif(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  int dim;
  if (!enif_get_int(env, argv[0], &dim) || dim <= 0 || dim % 64 != 0)
    return make_error(env, "dim_must_be_multiple_of_64");

  HdcVector *h =
      (HdcVector *)enif_alloc_resource(HDC_RESOURCE, sizeof(HdcVector));
  if (!h)
    return make_error(env, "out_of_memory");

  h->dim = dim;
  h->words = dim / 64;
  h->data = (uint64_t *)aligned_tensor_alloc(h->words * sizeof(uint64_t));
  if (!h->data) {
    enif_release_resource(h);
    return make_error(env, "out_of_memory");
  }
  memset(h->data, 0, h->words * sizeof(uint64_t));

  ERL_NIF_TERM term = enif_make_resource(env, h);
  enif_release_resource(h);
  return make_ok(env, term);
}

/** hdc_random(Dim, Seed) -> {ok, HdcRef}
 * Create random hypervector */
ERL_NIF_TERM hdc_random_nif(ErlNifEnv *env, int argc,
                                   const ERL_NIF_TERM argv[]) {
  (void)argc;
  int dim;
  unsigned long seed;
  if (!enif_get_int(env, argv[0], &dim) || dim <= 0 || dim % 64 != 0)
    return make_error(env, "dim_must_be_multiple_of_64");
  if (!enif_get_ulong(env, argv[1], &seed))
    return make_error(env, "invalid_seed");

  HdcVector *h =
      (HdcVector *)enif_alloc_resource(HDC_RESOURCE, sizeof(HdcVector));
  if (!h)
    return make_error(env, "out_of_memory");

  h->dim = dim;
  h->words = dim / 64;
  h->data = (uint64_t *)aligned_tensor_alloc(h->words * sizeof(uint64_t));
  if (!h->data) {
    enif_release_resource(h);
    return make_error(env, "out_of_memory");
  }

  vt_hdc_random(h->data, h->words, seed);

  ERL_NIF_TERM term = enif_make_resource(env, h);
  enif_release_resource(h);
  return make_ok(env, term);
}

/** hdc_bind(HdcA, HdcB) -> {ok, HdcC}
 * XOR binding: associates two concepts */
ERL_NIF_TERM hdc_bind_nif(ErlNifEnv *env, int argc,
                                 const ERL_NIF_TERM argv[]) {
  (void)argc;
  HdcVector *a = get_hdc(env, argv[0]);
  HdcVector *b = get_hdc(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_hdc");
  if (a->dim != b->dim)
    return make_error(env, "dim_mismatch");

  HdcVector *c =
      (HdcVector *)enif_alloc_resource(HDC_RESOURCE, sizeof(HdcVector));
  if (!c)
    return make_error(env, "out_of_memory");

  c->dim = a->dim;
  c->words = a->words;
  c->data = (uint64_t *)aligned_tensor_alloc(c->words * sizeof(uint64_t));
  if (!c->data) {
    enif_release_resource(c);
    return make_error(env, "out_of_memory");
  }

  vt_hdc_bind(a->data, b->data, c->data, a->words);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** hdc_similarity(HdcA, HdcB) -> {ok, Float}
 * Cosine-like similarity via Hamming distance */
ERL_NIF_TERM hdc_similarity_nif(ErlNifEnv *env, int argc,
                                       const ERL_NIF_TERM argv[]) {
  (void)argc;
  HdcVector *a = get_hdc(env, argv[0]);
  HdcVector *b = get_hdc(env, argv[1]);
  if (!a || !b)
    return make_error(env, "invalid_hdc");
  if (a->dim != b->dim)
    return make_error(env, "dim_mismatch");

  double sim = vt_hdc_similarity(a->data, b->data, a->words, a->dim);
  return make_ok(env, enif_make_double(env, sim));
}

/** hdc_permute(Hdc, Shift) -> {ok, HdcC}
 * Circular bit permutation for sequence encoding */
ERL_NIF_TERM hdc_permute_nif(ErlNifEnv *env, int argc,
                                    const ERL_NIF_TERM argv[]) {
  (void)argc;
  HdcVector *a = get_hdc(env, argv[0]);
  if (!a)
    return make_error(env, "invalid_hdc");

  int shift;
  if (!enif_get_int(env, argv[1], &shift))
    return make_error(env, "invalid_shift");

  HdcVector *c =
      (HdcVector *)enif_alloc_resource(HDC_RESOURCE, sizeof(HdcVector));
  if (!c)
    return make_error(env, "out_of_memory");

  c->dim = a->dim;
  c->words = a->words;
  c->data = (uint64_t *)aligned_tensor_alloc(c->words * sizeof(uint64_t));
  if (!c->data) {
    enif_release_resource(c);
    return make_error(env, "out_of_memory");
  }

  vt_hdc_permute(a->data, c->data, a->words, (size_t)shift);

  ERL_NIF_TERM term = enif_make_resource(env, c);
  enif_release_resource(c);
  return make_ok(env, term);
}

/** hdc_dim(Hdc) -> {ok, Int} */
ERL_NIF_TERM hdc_dim_nif(ErlNifEnv *env, int argc,
                                const ERL_NIF_TERM argv[]) {
  (void)argc;
  HdcVector *h = get_hdc(env, argv[0]);
  if (!h)
    return make_error(env, "invalid_hdc");

  return make_ok(env, enif_make_int(env, h->dim));
}
