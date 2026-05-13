# Landing Page

Static landing page published to GitHub Pages at
<https://gabrielmaialva33.github.io/viva_tensor/>.

## Structure

- `index.html` — single-page landing with inline `<style>`. No external CSS/JS.
- Links to `viva_tensor/index.html` (generated HexDocs HTML), `hexdocs.pm`,
  GitHub, the API guide and the paper.

## How it is built

`gleam docs build` produces HTML under `build/dev/docs/viva_tensor/`.
The `.github/workflows/docs.yml` workflow assembles a `_site/` directory:

```
_site/
├── index.html              # copied from docs/landing/index.html
└── viva_tensor/            # copied from build/dev/docs/viva_tensor/
    ├── index.html          # module overview
    ├── api.html            # docs/en/api.md (via gleam.toml)
    ├── paper.html          # docs/en/paper.md
    ├── stability.html
    ├── project-structure.html
    ├── ffi-architecture.html
    └── ...
```

That `_site/` directory is uploaded with `actions/upload-pages-artifact@v3`
and deployed via `actions/deploy-pages@v4`.

## Trigger

The publish workflow runs on:

- push to `main`
- tag pushes matching `v*`
- manual `workflow_dispatch`

The CI workflow (`ci.yml`) still builds docs as an artifact for PR validation;
it does not publish.

## Editing

To preview locally:

```bash
gleam docs build
mkdir -p _site/viva_tensor
cp docs/landing/index.html _site/index.html
cp -R build/dev/docs/viva_tensor/. _site/viva_tensor/
python3 -m http.server -d _site 8000
```

Open <http://localhost:8000/> to inspect the landing page and follow the
"HexDocs (local)" card into the generated module reference.

## Repository settings

The site requires **Pages source: GitHub Actions** in the repository settings
(`Settings → Pages → Build and deployment → Source: GitHub Actions`). No
`CNAME` file is shipped — the site is served from the default
`gabrielmaialva33.github.io/viva_tensor/` URL. If a custom domain is added
later, drop a `CNAME` file alongside `index.html` in `docs/landing/`.
