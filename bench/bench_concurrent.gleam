//// Benchmark de Concorrência - Onde Gleam BRILHA!
////
//// O poder do BEAM: milhões de processos leves
//// C/C++ libs são rápidas em single-thread, mas Gleam escala!
////
//// Run: gleam run -m viva_tensor/bench_concurrent

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor/tensor

pub fn main() {
  io.println(
    "╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║  CONCURRENCY BENCHMARK - Onde BEAM/Gleam BRILHA vs C/C++        ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝\n",
  )

  io.println(
    "C/C++ libs (Eigen, OpenBLAS, MKL) são rápidas em single-thread...",
  )
  io.println("Mas quantos tensors você processa em PARALELO? 🤔\n")

  // Test 1: Parallel tensor creation
  io.println("━━━ TEST 1: Criação Paralela de Tensors ━━━")
  bench_parallel_creation()

  // Test 2: Parallel reductions
  io.println("\n━━━ TEST 2: Reduções Paralelas ━━━")
  bench_parallel_reductions()

  // Test 3: Parallel dot products (like embedding similarity search)
  io.println("\n━━━ TEST 3: Similaridade em Batch (Embedding Search) ━━━")
  bench_parallel_similarity()

  // Test 4: Spawn many processes
  io.println("\n━━━ TEST 4: BEAM Process Spawning ━━━")
  bench_process_spawning()

  io.println(
    "\n╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║  CONCLUSÃO: BEAM escala horizontalmente, C/C++ escala vertical  ║",
  )
  io.println(
    "║  Para ML inference em produção: Gleam + Rust NIF = 🔥           ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝",
  )
}

fn bench_parallel_creation() {
  let counts = [100, 1000, 10_000]

  list.each(counts, fn(n) {
    // Sequential
    let #(seq_time, _) =
      timer_tc(fn() {
        list.range(1, n)
        |> list.map(fn(_) { tensor.random_uniform([100]) })
      })

    // Parallel using erlang:spawn and collect
    let #(par_time, _) =
      timer_tc(fn() {
        let parent = erlang_self()
        list.range(1, n)
        |> list.each(fn(_) {
          erlang_spawn(fn() {
            let result = tensor.random_uniform([100])
            erlang_send(parent, result)
          })
        })
        // Collect results
        collect_n(n)
      })

    let speedup = case par_time > 0 {
      True -> int.to_float(seq_time) /. int.to_float(par_time)
      False -> 1.0
    }

    io.println(
      "  "
      <> format_number(n)
      <> " tensors: seq="
      <> int.to_string(seq_time / 1000)
      <> "ms, par="
      <> int.to_string(par_time / 1000)
      <> "ms, speedup="
      <> float_to_string(speedup)
      <> "x",
    )
  })
}

fn bench_parallel_reductions() {
  // Create many tensors
  let tensors =
    list.range(1, 1000)
    |> list.map(fn(_) { tensor.random_uniform([1000]) })

  // Sequential sum of all
  let #(seq_time, _) =
    timer_tc(fn() { list.map(tensors, fn(t) { tensor.sum(t) }) })

  // Parallel sum
  let #(par_time, _) =
    timer_tc(fn() {
      let parent = erlang_self()
      list.each(tensors, fn(t) {
        erlang_spawn(fn() {
          let result = tensor.sum(t)
          erlang_send(parent, result)
        })
      })
      collect_n(1000)
    })

  let speedup = case par_time > 0 {
    True -> int.to_float(seq_time) /. int.to_float(par_time)
    False -> 1.0
  }

  io.println(
    "  1000 tensors x 1000 elementos: seq="
    <> int.to_string(seq_time / 1000)
    <> "ms, par="
    <> int.to_string(par_time / 1000)
    <> "ms, speedup="
    <> float_to_string(speedup)
    <> "x",
  )
}

fn bench_parallel_similarity() {
  // Simula busca de embedding: 1 query vs N documentos
  let query = tensor.random_uniform([512])
  let documents =
    list.range(1, 10_000)
    |> list.map(fn(_) { tensor.random_uniform([512]) })

  io.println("  Query vs 10K documents (512d embeddings):")

  // Sequential
  let #(seq_time, _) =
    timer_tc(fn() { list.map(documents, fn(doc) { tensor.dot(query, doc) }) })

  // Parallel batched (chunks of 100)
  let #(par_time, _) =
    timer_tc(fn() {
      let parent = erlang_self()
      let chunks = list.sized_chunk(documents, 100)
      let num_chunks = list.length(chunks)

      list.each(chunks, fn(chunk) {
        erlang_spawn(fn() {
          let results = list.map(chunk, fn(doc) { tensor.dot(query, doc) })
          erlang_send(parent, results)
        })
      })
      collect_n(num_chunks)
    })

  let speedup = case par_time > 0 {
    True -> int.to_float(seq_time) /. int.to_float(par_time)
    False -> 1.0
  }

  let throughput = case par_time > 0 {
    True -> 10_000.0 /. { int.to_float(par_time) /. 1_000_000.0 }
    False -> 0.0
  }

  io.println("    Sequential: " <> int.to_string(seq_time / 1000) <> "ms")
  io.println(
    "    Parallel:   "
    <> int.to_string(par_time / 1000)
    <> "ms (speedup: "
    <> float_to_string(speedup)
    <> "x)",
  )
  io.println(
    "    Throughput: " <> float_to_string(throughput) <> " queries/sec",
  )
}

fn bench_process_spawning() {
  io.println("  Quantos processos BEAM conseguimos spawnar?")

  let counts = [1000, 10_000, 100_000]

  list.each(counts, fn(n) {
    let #(time, _) =
      timer_tc(fn() {
        let parent = erlang_self()
        list.range(1, n)
        |> list.each(fn(_) { erlang_spawn(fn() { erlang_send(parent, 1) }) })
        collect_n(n)
      })

    let spawns_per_sec = case time > 0 {
      True -> int.to_float(n) /. { int.to_float(time) /. 1_000_000.0 }
      False -> 0.0
    }

    io.println(
      "    "
      <> format_number(n)
      <> " processos: "
      <> int.to_string(time / 1000)
      <> "ms ("
      <> float_to_string(spawns_per_sec)
      <> " spawns/sec)",
    )
  })

  io.println("")
  io.println(
    "  💡 Em C/C++ você precisaria de pthreads, mutex, condition vars...",
  )
  io.println(
    "  💡 Em Gleam: erlang_spawn() e pronto! Zero data races garantido.",
  )
}

// FFI - Direct Erlang calls
@external(erlang, "timer", "tc")
fn timer_tc(f: fn() -> a) -> #(Int, a)

@external(erlang, "erlang", "self")
fn erlang_self() -> Pid

@external(erlang, "erlang", "spawn")
fn erlang_spawn(f: fn() -> a) -> Pid

@external(erlang, "viva_tensor_ffi", "send_msg")
fn erlang_send(to: Pid, msg: a) -> a

@external(erlang, "viva_tensor_ffi", "collect_n")
fn collect_n(n: Int) -> List(a)

// Opaque types
type Pid

fn float_to_string(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 100.0)) /. 100.0
  float.to_string(rounded)
}

fn format_number(n: Int) -> String {
  case n >= 1_000_000 {
    True -> int.to_string(n / 1_000_000) <> "M"
    False ->
      case n >= 1000 {
        True -> int.to_string(n / 1000) <> "K"
        False -> int.to_string(n)
      }
  }
}
