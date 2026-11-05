# LCC "Uniform Light" (flat-field) — design notes

> Status: exploration only, no code yet (2026-11-05). The maintainer has approved an LCC feature. Scope: a
> flat-field correction like Capture One's LCC "Uniform Light" with an Amount slider. A reference shot through a
> diffuser gives a smoothed **luminance** gain map; the same gain multiplies R, G and B, so there is **no colour-cast
> correction**. Dust removal is out of scope.

**Naming gotcha before anything else:** every crate in Chromaforge is `lcc-*` / `lcc_*` (the legacy brand prefix,
[`AGENTS.md`](../AGENTS.md) "The name lives in `brand.toml` only"; e.g. `lcc_develop`, `lcc_raw`, `lcc_brand`). A
grep for "lcc" finds the brand, not lens cast calibration, and code named `lcc_*` would read as a crate. The code
should call the feature **flat field** (`flat_field`, `optics.flatField…`); "LCC" stays a UI/docs word.

## 1. How the Optics vignetting correction works today

**What it computes.** `Warp` in [`crates/pipeline/src/optics.rs`](../crates/pipeline/src/optics.rs) carries two
vignetting terms:

- Manual Vignetting: `gain = 2^(vig_stops · r^vig_power)` with `vig_stops = 1.5 · optics.vignetting / 100` and
  `vig_power = 1 + 6 · midpoint/100`, `r` = distance from the image centre over the half diagonal, capped at 1.5
  (`Warp::from_settings`, optics.rs:74–92; `Warp::gain`, optics.rs:237–251; `VIGNETTE_STOPS`, optics.rs:30).
- Profile vignetting: DNG `FixVignetteRadial` (`1 + k0 r² + … + k4 r¹⁰`) from the file or the lens database,
  scaled by `optics.profile_vignetting / 100` (`embedded_vignette_gain`, optics.rs:396–409; struct
  `EmbeddedVignette`, [`crates/develop/src/settings.rs`](../crates/develop/src/settings.rs):1162–1170). It is applied
  only when `optics.lens_profile` is on (optics.rs:84–90).

Both are one scalar per pixel, multiplied into R, G and B alike — exactly the shape a "Uniform Light" gain has.

**Where in the pipeline.** Stage 1, *geometry*, inside the single resample
([`crates/pipeline/src/lib.rs`](../crates/pipeline/src/lib.rs):7–14 lists the stage order). `sample_warped`
([`crates/pipeline/src/geometry.rs`](../crates/pipeline/src/geometry.rs):502–528) finds each output pixel's source
position `g`, samples it bilinearly and multiplies by `wp.gain(g)` (geometry.rs:521–524). The same gain is used by
generative-fill patches (`crates/pipeline/src/genpatch.rs`:75). The GPU renderer duplicates it: the warp's
parameters are packed into 65 floats (`crates/gpu/src/render.rs`:565–587, `debug_assert_eq!(p.len(), 65)`) and
`warp_gain` in `crates/gpu/src/wgsl/geom.wgsl`:137 is applied at geom.wgsl:194. CPU/GPU agreement is tested in
`crates/gpu/tests/it/equivalence.rs` (33 vignetting references).

A frame whose warp is the identity has no warp at all (`warp: (!warp.is_identity()).then_some(warp)`,
geometry.rs:94; `is_identity = !moves_pixels() && !has_gain()`, optics.rs:129–131), so a new gain must be part of
`has_gain()` (optics.rs:125–127) or it would be skipped on the Copy/Affine sampling paths.

Not to be confused with **Effects ▸ Vignette** (post-crop, creative): that runs in stage 5 on display-linear values
after the tone map (`crates/pipeline/src/finish.rs`:109–125, 861–880; GPU `finish.wgsl`:951).

**Is the data linear there?** Yes. The pipeline's input is "scene-referred, linear Rec.2020 … already EXIF-oriented"
(lib.rs:3–5). For a raw, `load_bytes_now` ([`crates/engine/src/files.rs`](../crates/engine/src/files.rs):715–903)
builds it as: decode → pixel repair on the mosaic (files.rs:738) → optional outlier suppression (740–746) →
`RawImage::develop` / `develop_binned_masked` (normalise, demosaic, `OpcodeList3`, default crop; files.rs:787–802) →
`finish` closure (files.rs:822–841): highlight rebuild → `CameraColourModel::apply` (white balance, camera matrix or
fitted look, DNG profile tables, baseline-exposure gain; files.rs:408–440) → fit to `max_edge` → orient. Exposure is
applied later, in stage 2 (lib.rs:19–21). So at the vignetting stage the values are linear Rec.2020 but already white
balanced, matrixed and scaled by baseline exposure, and not clipped (floats above 1.0 survive).

**Before or after highlight clipping/recovery?** After. Highlight reconstruction runs in the source load
(`rebuild_highlights` / `rebuild_highlights_masked`, files.rs:420–433, called at 826–829) on camera RGB with the
sensor clip `HIGHLIGHT_CLIP = 0.99` (files.rs:406), before white balance; the algorithm is in
`crates/raw/src/highlight.rs` (header 1–22, `clip_neutral` 28–31). The vignetting gain then multiplies the
reconstructed values. Process versions do not reach the source load ([`process-versions.md`](process-versions.md),
"Changes to how a file is decoded … the process doesn't reach that stage today").

**Coordinates.** Neither sensor nor output coordinates: the gain is evaluated at the **source position in the
user-oriented, EXIF-oriented, default-cropped source**, in pixels, normalised by its half diagonal (optics.rs:237–243;
the warp's `w × h` are the oriented source, optics.rs:34–38). So crop, straighten, Upright/Transform and flips don't
move it (the warp maps output → transformed → lens-corrected → source, optics.rs:3–11), and preview/export sizes
agree (resolution independence, AGENTS.md "Non-negotiables"). But it is not sensor-aligned:

- the active-area crop happens in `RawImage::normalized` and the default crop in `finish_demosaiced`
  (`crates/raw/src/lib.rs`:859–929), and non-square pixels are stretched (`stretched_height`, lib.rs:925–928);
- the source is EXIF-oriented at load (`img.into_oriented(raw.orientation)`, files.rs:837) and user orientation is
  applied in `Frame::with_lens` (geometry.rs:70–75).

File-embedded lens data is converted for this: `EmbeddedLens` is stored "converted to the oriented,
default-cropped image" (settings.rs:1112–1116) and re-expressed for the user orientation by `reorient_lens`
(optics.rs:411–446). A flat-field map recorded in sensor coordinates needs the same treatment.

## 2. Where linear, black-subtracted raw data exists, and could a gain map go there?

**Where.** `RawImage::normalized` (`crates/raw/src/lib.rs`:859–901) applies `OpcodeList1` on raw values, subtracts
each site's black, scales each site's white to 1.0, crops to the active area, then applies **`OpcodeList2` on the
normalised CFA data** (lib.rs:899). That is the earliest linear, black-subtracted point, still a mosaic and not white
balanced. After demosaic, `develop` gives camera RGB with white = 1.0, not white balanced (lib.rs:903–914), which is
what highlight reconstruction reads.

**There is already a gain-map mechanism here.** `crates/raw/src/opcodes.rs` parses, writes and applies DNG
`GainMap` (opcodes.rs:88, parse 404, `gain_at` 550, applied in `apply_list` 609–635), including binning/cropping of
its area (opcodes.rs:135–177). A DNG flat field is exactly a `GainMap` in `OpcodeList2`. (Not to be confused with
`crates/raw/src/gaintable.rs`, ProRAW's tone-mapping `ProfileGainTableMap`.)

**Could ours go there?** It could, but it costs more than it gains for an achromatic map:

- **Amount would re-decode the raw.** Develop settings that change the source go through `LoadOptions::of`
  (`crates/engine/src/media.rs`:179–188) and `DecodedSource::serves` (media.rs:277–285), as `sensor.pixelFix` and
  `detail.nr_outliers` do; any change drops the decoded source. A slider drag would then mean a full decode per step,
  far past the 16 ms slider budget (AGENTS.md "Testing & performance").
- **Previews lose the fast path.** `develop_binned` falls back to a full `normalized()` whenever `OpcodeList1/2` are
  non-empty (`crates/raw/src/binned.rs`:70), so injecting a map there slows every thumbnail and preview unless the
  binned path learns the map.
- **Denoise and highlights must follow.** The AI-denoised twin is made from `raw.normalized()`
  (`crates/engine/src/denoise.rs`:1245) and read back as a cached product (files.rs:697–713), so its key would need
  the map; highlight reconstruction's clip test (0.99) would need dividing by the gain, or brightened corners read as
  clipped.
- **It buys nothing for a uniform gain.** Everything between the mosaic and stage 1 is linear and per-pixel except
  highlight reconstruction and the colour tables (`CameraColourModel::apply`, files.rs:436–440). A scalar gain
  commutes with white balance and the 3×3 matrix; the hue/sat tables are indexed by hue and saturation, which a
  scalar gain leaves alone (a DNG `LookTable` with a value axis is the exception). Raw-stage placement matters for a
  **colour-cast** LCC (per-channel gains, which the sensor's colour crosstalk produces); that is out of scope here.

So the current stage is not "too late" for Uniform Light. The open question is only highlights (see §7).

## 3. How develop settings are stored and versioned

**Schema.** `DevelopSettings` (settings.rs:271–347) is one serde struct, `#[serde(default)]` everywhere: missing
fields take defaults, unknown ones are ignored (settings.rs:1–4); `SCHEMA_VERSION = 1` (settings.rs:9). Field names
are snake_case in JSON (`optics.profile_vignetting`); control ids are camelCase (`optics.profileVignetting`).
Optics fields: settings.rs:1050–1107. Fields added later avoid changing old settings' JSON with
`skip_serializing_if = "…::is_default"` (`hdr`, `skin`, `retouch`, `sensor`; settings.rs:286–336), because
`DevelopSettings::hash64` hashes the JSON (`crates/develop/src/lib.rs`:99–106) and that hash keys preview caches and
the catalog's content-addressed `develop_settings` table ([`catalog.md`](catalog.md) "develop settings are
content-addressed").

**Process versions.** `ProcessVersion` / `Process` (settings.rs:146–270), now V1…V7, LATEST = V7
(settings.rs:216–269); rules and table in [`process-versions.md`](process-versions.md): a change to how *existing*
settings render needs a new process; a new setting that is neutral by default does not. A flat field at Amount 0 / no
reference is neutral, so it needs no process.

**Catalog format.** The catalog format is **16** now, not 13: `VERSION = 16` in
`crates/catalog/src/journal.rs`:91, table at journal.rs:43–60 (13 was `Meta.gps_removed`; [`catalog.md`](catalog.md)
line 112 still says 13 and is stale). The rule (journal.rs:62–66): bump `VERSION` when adding a serialized field "to …
develop settings that an older build would misread or silently drop". In practice recent develop fields (`sensor`,
`hdr`, `skin`) shipped without a row; see open questions.

**XMP sidecars.** `lc:settings` holds the complete `DevelopSettings` JSON (exact round trip;
[`xmp-interop.md`](xmp-interop.md):32, `crates/engine/src/sidecar.rs`:6, 29, 244), so a new field travels
automatically. `crs:` fields are mapped approximately in `crates/engine/src/crs.rs` (`VignetteAmount` →
`optics.vignetting`, crs.rs:479–480); Adobe has no flat-field equivalent to map. Pixel-filter stacks are the one
thing written under their own namespace instead (sidecar.rs:749).

**Copy / paste / sync / presets.** `SettingsGroup` in `crates/develop/src/presets.rs`: the Optics group copies the
whole `optics` object (presets.rs:131–134), and `OPTICS_ITEMS` lists sub-items that can be ticked one by one
(presets.rs:228–245). The process is in no group (presets.rs:9–11; process-versions.md "Copy, paste, sync and
presets").

**What a new setting needs** (following `sensor.pixelFix` and `optics.vignetting`):

1. The field with a neutral `Default`, ideally skipped while default, in `Optics` (settings.rs:1050–1107).
2. A control spec for the numeric Amount in `crates/develop/src/controls.rs` (the table at ~295–347; e.g.
   `"optics.vignetting" => optics.vignetting, "Vignetting", Optics, -100, 100, 0, 1, 0, Centered;` at 320). Every
   slider is a control spec (AGENTS.md "Everything is a command"); clamping of hostile values is tested in
   `crates/develop/src/lib.rs`:544–546.
3. A sub-item in `OPTICS_ITEMS` (presets.rs:228–245) so it can be copied without the other optics settings.
4. The reference itself (not a number) set through `develop.merge` like the lens picker's `optics.lens`
   (`ChosenLens`, settings.rs:1072–1090) — plus commands for making and listing references (engine command registry,
   `crates/engine/src/cmd/`), so UI, CLI, control channel and MCP share them.
5. The renderer: CPU (`Warp`, optics.rs) and GPU (`render.rs` warp params, `geom.wgsl`), with the equivalence test.
6. Catalog `VERSION` bump if the maintainer applies journal.rs's rule to develop fields; `RENDER_CACHE_VERSION`
   (`crates/engine/src/media.rs`:65, now 77) only if existing renders change (they shouldn't).

## 4. The Optics panel in each layout, and locale strings

- **Desktop (Lightroom-style right panel):** `panels/right.rs`:91 calls `edit::show_controls`, which draws the
  section `section(app, ui, &d, "optics", "Lens Corrections", …)` in
  [`crates/ui-egui/src/panels/edit.rs`](../crates/ui-egui/src/panels/edit.rs):418–498: Remove CA and Enable Lens
  Corrections checkboxes (`develop.merge`), lens note and picker, profile sliders when enabled (435–437), the "New
  photos start from" default (442–468), then **Manual**: `optics.distortion`, `optics.vignetting`,
  `optics.vignettingMidpoint` (470–474), and the Defringe flyout (476–496). Sliders are `control()` / `control_as()`
  (edit.rs:41–50), built from the control specs. Menu entry `section.optics` = Cmd+6 (`crates/ui-egui/src/menus.rs`:199).
- **Modern (touch) layout:** [`crates/ui-egui/src/modern_dock.rs`](../crates/ui-egui/src/modern_dock.rs) — the
  Optics dock group's pill sliders are a fixed list `&["optics.distortion", "optics.vignetting", "optics.caRed",
  "optics.caBlue"]` (modern_dock.rs:130); its "more" button opens the same Edit-panel sections, `("optics", "Lens
  Corrections")` and `("geometry", "Transform")` (modern_dock.rs:108), via `edit::show_section` (modern_dock.rs:472);
  a long press on any `optics.*` slider opens the `optics` section (modern_dock.rs:139–148). Docs:
  [`android.md`](android.md):15.
- There are two layouts, Desktop and Touch/Modern (Settings ▸ General ▸ Layout; [`android.md`](android.md):15).
  `panels/classic.rs` is not a third one: it is the Lightroom Classic–style panel stack of the Library side
  (Navigator, Folders, Quick Develop…; classic.rs:1–5), and the Desktop Develop column (`panels/right.rs`:54–93, with
  Classic's Previous / Sync… bar from `classic_tools.rs`) shows the same `edit::show_controls`. So there is one Optics
  section to change (edit.rs) plus the Modern pill list (modern_dock.rs:130), if the Amount should be a pill too.
- UI tests touching Optics: `tests_develop_panels.rs`, `tests_panels.rs`, `tests_compact_layout.rs`,
  `tests_solo_sections.rs`, `tests_lens_db.rs` in `crates/ui-egui/src/`.

**Locale strings.** Text goes through `crate::i18n::tr("…")`; keys are the English strings
([`localization.md`](localization.md) "How a message is found"). Upstream's catalogs are
`crates/ui-egui/locales/<code>.json`; **Chromaforge's own strings go in `crates/ui-egui/locales/fork/<code>.json`**,
merged over upstream's (`crates/ui-egui/src/i18n_fork.rs`:1–3). `tests_i18n_coverage.rs` fails when a fork string is
missing from a complete language (every language except the `PARTIAL` list: pl, cs, it, ka, sv, fi, id, vi;
tests_i18n_coverage.rs:23–28); the fast gate also checks the fork catalogs are sorted ([`ci.md`](ci.md)). Control
labels ("Vignetting") are translated the same way (e.g. `locales/de.json`:1589).

## 5. Capture One catalog import

`crates/engine/src/capture_one_catalog.rs` (1001 lines) reads library data only: "Develop settings are not read yet"
(capture_one_catalog.rs:9–11). `read_with_progress` (279–) loads `ZENTITIES`, `ZIMAGE`, `ZPATHLOCATION`,
`ZVARIANT`, `ZVARIANTMETADATA`, `ZKEYWORD`, collections and `ZDOCUMENTCONTENT` (291–303). The only per-variant
layer lookup is metadata: rating, colour tag, IPTC from the `ZVARIANTMETADATA` rows of the variant's
`ZCOMBINEDSETTINGS`, `ZADJUSTMENTLAYER` and `ZDEFAULTLAYER` (362–366). **Nothing in the importer reads
`ZVARIANTLAYER` or anything LCC-like**; a grep for `ZSTYLESOURCE`, `ZVARIANTLAYER`, `.lcc` or flat field in `crates/`
finds nothing.

Outside the code (from the `capture-one-develop` investigation and the user's catalog, read-only):

- In `ZVARIANTLAYER`, LCC is applied as a *style* layer (entity 19 `VariantStyleLayer`, `ZSTYLESOURCE = 'LCC'`).
- The catalog's `Adjustments/LCC/` folder holds the profiles as `<32-hex>.lcc` files (24 in the user's main catalog,
  12–520 KB). Their header is a tagged binary container naming "CaptureOne LCC", a version, camera make, model and
  **body serial**, a raw hash and a "Chroma" block. The format is undocumented; reading it would be black-box
  analysis of the user's own files (allowed for data, AGENTS.md "Clean-room sources"), and is a separate task.

The develop import is paused until this tool exists, so it can bring LCC across in one pass.

## 6. Tests, CI and PR / changelog conventions

- **Tests** ([`AGENTS.md`](../AGENTS.md) "Testing & performance"): unit/property tests next to the code (crates use
  `tests*.rs` siblings, e.g. `crates/pipeline/src/tests_geometry.rs`, `tests_window.rs`); behaviour and invariants,
  no change-detector or golden-image tests; procedural images from `lcc-scenes`, never committed media; hostile-input
  no-panic checks for anything parsed (an imported reference or `.lcc`). CPU/GPU parity in
  `crates/gpu/tests/it/equivalence.rs`. Process rules have `crates/develop/src/tests_process.rs`. End-to-end:
  `apps/cli/tests/`, `crates/mcp/tests/it/e2e.rs`; SIL harness `apps/cli/tests/sil.rs`, `cargo xtask sil`
  ([`testing-sil.md`](testing-sil.md)). Tests that can't run here use `lcc_testkit::skip!`.
- **Never crash:** no `unwrap`/`expect`/`panic!`/indexing on input-derived numbers in non-test code, enforced by
  clippy (AGENTS.md "Never crash"). A gain map from a file needs `get()`, checked sizes and capped allocations.
- **CI** ([`ci.md`](ci.md)): PRs run `cargo xtask ci --fast` (fmt, brand, docs, notes, i18n sort, clippy
  `-D warnings`, parity refs, layers, assets, deny, wasm and Android checks) plus SIL; the full `cargo xtask ci`
  runs before releases. One `ci` per machine (`xtask/src/ci_lock.rs`). Per commit: `--fast` + `cargo test -p` of the
  touched crates. Perf: `cargo xtask bench`, `<PREFIX>_PROFILE=1`.
- **Commits / PRs:** one task id per commit (`M2.3: …`); PR title starts with the checklist id or `Issue #n:`
  ([`CONTRIBUTING.md`](../CONTRIBUTING.md) "Open a pull request"); template `.github/PULL_REQUEST_TEMPLATE.md` (Problem,
  Change, How it was checked, "Existing edits and libraries", "Shared numbers taken" — catalog `VERSION`,
  `RENDER_CACHE_VERSION`, `LOOK_VERSION`, process). A **Sources:** line for every algorithm (here: "own design" / the
  DNG spec's GainMap). Branch from latest `main`, merge (not rebase) `origin/main` before pushing.
- **Parity docs in the same commit:** the row for this belongs next to `LR-EDIT-OPTICS-MANUAL`
  ([`parity-checklist.md`](parity-checklist.md):299) — a new beyond-Lightroom row; `docs/gaps.md` if listed;
  `cargo xtask parity` checks cited ids and paths.
- **Release notes:** `docs/whats-new/<YYYY-MM-DD>-<slug>.md` (one `### Title` + bullets; `docs/whats-new/README.md`)
  and `docs/changelog/unreleased/<YYYY-MM-DD>-<slug>.md` (one `## Develop` section + bullets with `PR #n`;
  `docs/changelog/unreleased/README.md`). Never edit `docs/whats-new.md` or a released changelog;
  `cargo xtask notes --check` enforces it.
- **Docs** name the fork "Chromaforge" and never write binary names or the env prefix outside `.md.in` templates
  (AGENTS.md "The name lives in `brand.toml` only"; `cargo xtask brand check`).

## 7. Open questions

1. **Highlights in dark corners.** With the gain after highlight reconstruction, a corner that clipped on the sensor
   is rebuilt, then brightened past the rest of the frame's clip level. Is that acceptable (it matches today's
   vignetting slider), or should reconstruction see the gain? Needs a test on a real flat-field + clipped-corner raw.
2. **Where reference profiles live.** Options: in the settings JSON (a coarse grid, e.g. 64 × 43 f32 ≈ 11 KB, bloats
   every edited photo and preset; travels in `lc:settings`), or a per-body store referenced by id, like
   `crates/engine/src/sensor_maps.rs` (`<config>/sensor-maps`, per body serial) and camera profiles
   (`crates/engine/src/camera_profiles.rs`, `dcp:<id>`). A store keeps settings small but a sidecar or copied
   library then carries a dangling id. What happens on the web build, which has no config folder?
3. **Grid and smoothing.** Grid size, smoothing radius, and how the reference is normalised (centre = 1.0, or
   brightest = 1.0 so the map only brightens?). Amount: 0–100 linear in stops (`gain^(amount/100)`) or in gain?
4. **Coordinates of the map.** Store it relative to the active area (sensor), then convert per photo for default
   crop, pixel aspect, EXIF and user orientation, as `reorient_lens` does; and refuse a reference whose raw size
   doesn't match (sensor maps refuse other models/sizes the same way).
5. **Interaction with lens-profile vignetting.** A flat field already removes lens fall-off; should applying one
   switch Profile Vignetting off (C1 does not touch its own lens correction), or multiply both?
6. **Catalog format bump.** journal.rs's rule says a new develop field older builds would drop needs a `VERSION`
   bump (17), but `sensor`, `hdr` and `skin` shipped without one. Which does the maintainer want here?
7. **Making a reference.** A command on a selected raw (`flatField.create`?), batch per lens/aperture, and how the
   C1 importer maps an `.lcc` (decode its grid, or ask the user to re-shoot)?
8. **Copy semantics.** Should the reference be in the Optics copy group (so syncing a shoot copies it) — C1's own
   workflow copies LCC across the shots of one lens and aperture.

## 8. Recommendation: where the flat-field stage goes

**Put the gain in stage 1, inside `Warp::gain`** (`crates/pipeline/src/optics.rs`:237), as a third factor next to
manual and profile vignetting, sampled bilinearly from a smoothed grid at the source position:

- it is where Chromaforge already applies achromatic fall-off corrections, linear and before every spatial and
  tonal stage (noise reduction, local tone, masks), so masks and Auto see the evened-out picture;
- the Amount slider stays cheap: the decoded source is reused and only the geometry resample reruns, like dragging
  Vignetting today (no re-decode, no lost binning fast path, no denoise-product key);
- resolution independence, crops, Upright and windowed renders come for free from the existing framing
  (geometry.rs:502–528), and generative fill picks it up through the same `gain()` (genpatch.rs:75);
- for a same-gain-on-all-channels map the result equals a raw-stage gain except for highlight reconstruction
  (open question 1).

What it requires: extend `has_gain()` so the warp isn't dropped as identity (optics.rs:125–131); carry the grid on
`SourceInfo` / `Frame` already converted to the oriented, default-cropped frame (like `EmbeddedLens`); add it to the
GPU path (a small storage buffer or texture beside the 65-float params in `render.rs`:565–587, sampled in
`geom.wgsl` `warp_gain`) with an equivalence test; and keep the settings field neutral and skipped while default so
no existing photo's JSON, hash or render changes (no new process).

Revisit the raw stage (`OpcodeList2`-style `GainMap` in `RawImage::normalized`, `crates/raw/src/lib.rs`:899) only
if colour-cast correction comes into scope later; the GainMap opcode code in `crates/raw/src/opcodes.rs` is the
starting point then.

## 9. Phase 1: measuring maps (2026-11-05)

**Amount and what we validate against.** Recent Capture One versions overcorrect at LCC amount 100 (users set
about 43 to look right), while older versions were correct at 100. Chromaforge does not emulate either:

- The map is measured on linear data and applied on linear data, so amount 1.0 is the physically correct full
  correction. `DEFAULT_FLAT_FIELD_AMOUNT = 1.0`, and the range is 0 to 1.2 (Capture One's slider).
- The amount scales the correction in EV: `gain^amount`.
- A map is checked against a second reference of the same setup (`--check` below), not against Capture One's
  output.

**A map is valid only for an identical active area.** It is a grid over the sensor's active area, unrotated
(`FlatFieldMap::sensor`). Phase 2 must refuse a photo whose active area differs. This includes the same body in
another image area: a Z 7 shot in 5:4 records 6912 × 5520 px instead of 8288 × 5520. It also includes another body
of the same lens, even a near-identical sensor: Z 7 is 8288 px wide, Z 8 is 8280. `--check` always refuses a size
mismatch.

**What landed** (`crates/raw/src/flat_field.rs`, CLI `apps/cli/src/flat_field.rs`):

- `FlatFieldMap` (serde, grid of gains ≥ 1, never pre-scaled) and `FlatFieldSource` (camera, serial, lens, focal
  length, aperture, SHA-1).
- `build`:
  - Takes the green samples of `RawImage::normalized`, which already has any DNG `OpcodeList2` GainMap applied, so
    the map measures the fall-off that remains.
  - Per-cell trimmed mean; any cell with more than 1 % clipped green samples refuses the reference (`Clipped`,
    with the cell and percentage).
  - log2, 3 × 3 median and Gaussian, both with odd-reflection padding; normalised to the brightest cell.
- Sampling (`gain_at`, `ev_at`) is bilinear in log space, with linear extrapolation over the last half cell to the
  frame edge.
- `SensorFrame` maps between `Warp::gain`'s coordinates (EXIF and user orientation, default crop, pixel aspect, any
  fitted size) and the sensor.
- `fit_centre` fits `a + b·d² + c·d⁴` with a coarse-to-fine centre search and reports the offset and RMS.
- `check` reports the residual against the median corrected cell. It passes at max |residual| < 0.05 EV and
  peak-to-peak < 0.1 EV.
- `<cli> flat-field REF.raw [--check B.raw [--force]] [--save MAP.json] [--json]` prints corners, the radial table,
  the optical centre, and the lens database's vignetting evaluated as `Warp::gain` does.

Nothing is in the pipeline, the settings or the UI yet.

**First real references** (17 NEFs, Z 7 and Z 8):

- All measure without clipping. Corner fall-off ranges from 0.8 EV (500 mm f/5.6) to 3.6 EV (24 mm f/1.8).
- The only same-setup pair with an identical active area (28 mm f/2.8 on the Z 7) fails `--check`: 0.5 EV
  peak-to-peak, with centres 4 % apart and fit RMS around 0.05 EV. One of the two shots has uneven light (a sky
  gradient or the diffuser).
- The other same-setup pairs differ in image area (5:4 vs 3:2), and the Z 7 / Z 8 pairs differ in width, so they
  can't be compared under the identical-area rule.
- Better references are needed: two diffuser shots per setup at the same image area.
