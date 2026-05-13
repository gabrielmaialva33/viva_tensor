# Development Modules

This directory contains Gleam modules that are useful while developing
`viva_tensor`, but are not part of the supported package API.

## Layout

| Path                          | Purpose                                                                                                                   |
|:------------------------------|:--------------------------------------------------------------------------------------------------------------------------|
| `viva_tensor/bench/`          | Gleam benchmark entrypoints used by Makefile targets such as `make bench`, `make bench-rtx`, and `make bench-regression`. |
| `viva_tensor/examples/`       | Runnable examples and demos for maintainers.                                                                              |
| `viva_tensor/benchmark.gleam` | Shared development benchmark helpers.                                                                                     |

These modules are intentionally kept out of `src/` so Hex users do not see them
as stable library modules. Public examples should import the root facade with
`import viva_tensor as t`.
