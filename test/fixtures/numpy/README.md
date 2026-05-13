# NumPy Reference Fixtures

This directory holds JSON fixtures that pin viva_tensor's numerical output
against NumPy. Each fixture is a small, self-contained test case for one
operation (`add`, `matmul`, `exp`, ...). The Gleam test suite consumes them
via `test/reference_test.gleam` and asserts results match within an
`np.allclose`-style tolerance.

## Layout

```
test/fixtures/numpy/
  gen_reference.py        # generator script (run to regenerate fixtures)
  <op>/<case>.json        # one fixture per op + shape combo
```

Each fixture has the schema:

```json
{
  "op": "add",
  "case": "vec4",
  "inputs":  [{"shape": [4], "data": [...]}, ...],
  "output":  {"shape": [4], "data": [...]},
  "tolerance": {"rtol": 1e-7, "atol": 1e-9}
}
```

Scalar outputs (e.g. `sum`, `mean`) use `shape: []` and a single-element
`data` array so the decoder shape is uniform.

## Regenerating

NumPy is the only Python dependency. Pick whichever runner you prefer:

```bash
# preferred — uses the workspace toolchain
uv run python test/fixtures/numpy/gen_reference.py

# fallback — system python3 with numpy on the path
python3 test/fixtures/numpy/gen_reference.py
```

If NumPy is not installed in the active environment:

```bash
uv add numpy --dev          # uv-managed projects
pip install --user numpy    # plain python3 fallback
```

The script overwrites all fixtures it owns and prints the relative paths it
touched. Commit the regenerated JSON files alongside any change to the
generator.

## Adding a new case

1. Add a new `yield "<op>", "<case>", [inputs...], output` line in
   `gen_reference.py`.
2. Run the generator.
3. Wire a matching call in `test/reference_test.gleam` (or extend the
   relevant op-group test).

Keep cases small (shapes in single digits) so the JSON stays under 1 KB and
the test suite stays fast.
