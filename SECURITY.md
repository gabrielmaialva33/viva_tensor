# Security Policy

viva_tensor includes native code and GPU/CPU acceleration paths. Security reports are welcome, especially around NIF memory safety, resource lifetime, bounds checks, CUDA/MKL integration, unsafe pointer use, build scripts, and supply-chain assumptions.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| `1.0.x` | Yes |
| `2.x` | No, treated as premature development versioning |
| `< 1.0` | No |

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability.

Use GitHub Security Advisories if available for this repository. If advisories are unavailable, contact a maintainer privately through GitHub and avoid sharing exploit details publicly until a fix is available.

Include as much detail as possible:

- Affected version or commit.
- Operating system, Erlang/OTP, Gleam, Zig, CUDA, MKL, and GPU/CPU details.
- Minimal reproducer.
- Expected and actual behavior.
- Crash logs, sanitizer output, or backtrace if available.
- Whether the issue is memory corruption, out-of-bounds access, denial of service, incorrect result, or build-chain risk.

## Response Expectations

Maintainers will try to acknowledge valid reports promptly, reproduce the issue, and coordinate a fix before public disclosure. Timing depends on severity, hardware requirements, and whether the issue is in Gleam, Erlang, Zig/C, CUDA, MKL, or an external dependency.

## Native Code Notes

Native code changes should be reviewed with extra care. Prefer explicit shape checks, resource ownership clarity, graceful fallback when acceleration is unavailable, and tests that cover failure paths as well as fast paths.
