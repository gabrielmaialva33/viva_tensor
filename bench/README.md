# Benchmark Suite

Ferramentas para benchmarking estatístico rigoroso de viva_tensor.

## Uso Rápido

```bash
# Benchmark completo (30 runs, Bootstrap CI 95%)
python3 bench/benchmark.py

# Análise estatística (requer R)
Rscript bench/analysis.R

# Runner automatizado (CI/CD)
./bench/run_benchmarks.sh
```

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `benchmark.py` | Benchmark principal - Bootstrap CI, Mann-Whitney U, Cliff's delta |
| `analysis.R` | Análise estatística avançada com ggplot2 |
| `benchmark_compare.py` | Comparativo simples entre bibliotecas |
| `run_benchmarks.sh` | Runner para CI/CD |
| `bench_*.erl` | Benchmarks Erlang específicos |
| `bench_*.bat` | Scripts Windows |

## Output

```
bench/
├── data/           # JSON com dados brutos (gitignored)
└── reports/        # Gráficos e relatórios (gitignored)
```

## Metodologia

Seguindo [Kalibera & Jones (2013)](https://dl.acm.org/doi/10.1145/2400682.2400691):

1. **Warmup**: 5 runs descartados
2. **Timed**: 30 runs cronometrados
3. **Outliers**: Remoção via IQR (1.5×)
4. **CI**: Bootstrap BCa 95% (10.000 resamples)
5. **Testes**: Mann-Whitney U (não-paramétrico)
6. **Effect Size**: Cliff's delta

## Ambiente

```bash
# Linux/WSL
export MKL_NUM_THREADS=24
export OMP_NUM_THREADS=24

# Windows
set MKL_NUM_THREADS=24
set OMP_NUM_THREADS=24
```
