# Landing Page

Static landing page published to GitHub Pages at
<https://gabrielmaialva33.github.io/viva_tensor/>.

## Structure

- `index.html` — English single-page landing (default).
- `pt-br.html` — Portuguese (Brazilian) translation.
- `zh-cn.html` — Simplified Chinese translation.

All three share the same inline `<style>` block. No external CSS/JS. Each page
carries a language switcher chip in the top-right corner.

Links target:
- generated HexDocs (`viva_tensor/index.html`)
- published `hexdocs.pm/viva_tensor`
- GitHub repository, LLM API guide and paper
- `CHANGELOG.md`

## How it is built

`gleam docs build` produces HTML under `build/dev/docs/viva_tensor/`.
The `.github/workflows/docs.yml` workflow assembles a `_site/` directory:

```
_site/
├── index.html              # copied from docs/landing/index.html (EN)
├── pt-br.html              # copied from docs/landing/pt-br.html
├── zh-cn.html              # copied from docs/landing/zh-cn.html
├── README.md               # copied (not served)
└── viva_tensor/            # copied from build/dev/docs/viva_tensor/
    ├── index.html          # module overview
    ├── llm.html            # docs/en/api/llm.md (via gleam.toml)
    ├── inference.html
    ├── tensor.html
    ├── paper.html
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
cp -R docs/landing/. _site/
cp -R build/dev/docs/viva_tensor/. _site/viva_tensor/
python3 -m http.server -d _site 8000
```

Open <http://localhost:8000/> (EN), <http://localhost:8000/pt-br.html>
(PT-BR), or <http://localhost:8000/zh-cn.html> (ZH-CN) to inspect and
follow the "HexDocs (local)" card into the generated module reference.

Adding a new locale: copy `index.html` to `<locale>.html`, swap the
`<html lang="...">` attribute, translate visible text, and add the locale
chip to the `.lang-switcher` block in every page.

## Repository settings

The site requires **Pages source: GitHub Actions** in the repository settings
(`Settings → Pages → Build and deployment → Source: GitHub Actions`). No
`CNAME` file is shipped — the site is served from the default
`gabrielmaialva33.github.io/viva_tensor/` URL. If a custom domain is added
later, drop a `CNAME` file alongside `index.html` in `docs/landing/`.
