//! `<cli> calibrate-lightroom`: fit a camera's colour model and named looks to Lightroom's (or, with `--editor`,
//! Capture One's, darktable's, RawTherapee's or DxO PhotoLab's) exports of the
//! contributor's own raws (`lcc_colourfit`), and the developer step that adds a contributed file to the repository.
//!
//! - `preset -o FILE`: the baseline Lightroom preset contributors apply before exporting.
//! - `fit --raws DIR… --exports DIR… [-o OUTDIR] [--contributor NAME] [--lens MODEL] [--dry-run]`: pair, sample,
//!   fit; writes the calibration JSON (the file to send) and one `.dcp` per look for the contributor's own local
//!   testing. `--lens` makes the fit one lens's (the photos' Exif lens model, e.g. one of a phone's cameras).
//! - `add FILE [--repo DIR] [--contributor NAME] [--lens MODEL] [--replace]`: validate a contribution and
//!   commit-ready write it into `crates/camdb/data/calibrations/` with its `assets/ATTRIBUTION.md` row. A file for
//!   the same camera, lens and editor is kept unless `--replace` (issue #1015); another lens's fit gets its own file.
//! - `list`: the calibrations built into this binary.
//! - `probe -o DIR (--like RAW | --make M --model M)`: synthetic probe DNGs and slider sweeps for Lightroom to render;
//!   `probe-check --probes DIR --exports DIR`: read them back (`lcc_colourfit::probe`).
//! - `verify --raws DIR… --exports DIR… [--report DIR] [--json] [--size N] [--max N]`: the app's render of edited
//!   photos against their edited exports, ΔE00 per photo and per group of controls used (`lcc_colourfit::verify`).
//!
//! Sources: own design; DNG specification 1.7 chapter 6 (camera profile tags) for the `.dcp` writer.

use std::path::{Path, PathBuf};

use lcc_camdb::calibration::{self, Calibration};
use lcc_colourfit::dataset::{self, CollectOptions, ExportSet};
use lcc_colourfit::fit::{self, FitOptions};
use lcc_raw::ColorData;

/// Where `add` writes calibrations, relative to the repository root.
const DATA_DIR: &str = "crates/camdb/data/calibrations";

pub fn run(args: &[String]) -> Result<(), String> {
    let rest = args.get(1..).unwrap_or_default();
    match args.first().map(String::as_str) {
        Some("preset") => preset(rest),
        Some("fit") => fit_cmd(rest),
        Some("add") => add(rest),
        Some("list") => list(),
        Some("verify") => verify_cmd(rest),
        Some("probe") => probe_cmd(rest),
        Some("probe-check") => probe_check_cmd(rest),
        _ => Err("calibrate-lightroom needs `preset`, `fit`, `add`, `list`, `verify`, `probe` or `probe-check` (see --help)".into()),
    }
}

fn take<'a>(args: &'a [String], i: &mut usize, flag: &str) -> Result<&'a str, String> {
    *i += 1;
    args.get(*i).map(String::as_str).ok_or_else(|| format!("{flag} needs a value"))
}

fn verify_cmd(args: &[String]) -> Result<(), String> {
    use lcc_colourfit::verify::{self, VerifyOptions};
    let (mut raw_dirs, mut export_dirs) = (Vec::new(), Vec::new());
    let mut report_dir: Option<PathBuf> = None;
    let mut json = false;
    let mut opts = VerifyOptions::default();
    let mut list: Option<bool> = None; // Some(true) = raws, Some(false) = exports
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--raws" => list = Some(true),
            "--exports" => list = Some(false),
            "--report" => report_dir = Some(take(args, &mut i, "--report")?.into()),
            "--json" => json = true,
            "--size" => {
                let v = take(args, &mut i, "--size")?;
                opts.compare.long_edge = v.parse().map_err(|_| format!("--size: `{v}` is not a number of pixels"))?;
            }
            "--max" => {
                let v = take(args, &mut i, "--max")?;
                opts.max_photos = v.parse().map_err(|_| format!("--max: `{v}` is not a number of photos"))?;
            }
            a if a.starts_with('-') => return Err(format!("unknown option `{a}`")),
            a => match list {
                Some(true) => raw_dirs.push(PathBuf::from(a)),
                Some(false) => export_dirs.push(PathBuf::from(a)),
                None => return Err(format!("`{a}`: put folders after --raws or --exports")),
            },
        }
        i += 1;
    }
    if raw_dirs.is_empty() || export_dirs.is_empty() {
        return Err("verify needs --raws DIR… and --exports DIR… (the edited exports, searched with their subfolders)".into());
    }
    let mut raws = Vec::new();
    for d in &raw_dirs {
        raws_in(d, &mut raws, 0);
    }
    raws.sort();
    raws.dedup();
    if raws.is_empty() {
        return Err("no raw files found in the --raws folders".into());
    }
    let exports: Vec<ExportSet> = export_dirs.iter().map(|d| ExportSet { dir: d.clone() }).collect();
    let mut progress = |done: usize, total: usize, name: &str| {
        if done < total {
            eprintln!("calibrate-lightroom verify: {}/{total} {name}", done + 1);
        }
    };
    let report = verify::verify(&raws, &exports, &opts, &mut progress).map_err(|e| e.to_string())?;
    if let Some(dir) = &report_dir {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for p in &report.photos {
            let Some(rep) = &p.report else { continue };
            let rgb = lcc_colourfit::compare::map_rgb8(rep);
            let (Ok(w), Ok(h)) = (u32::try_from(rep.width), u32::try_from(rep.height)) else { continue };
            let img = lcc_codecs::EncodeImage::new(w, h, 3, lcc_codecs::Samples::U8(&rgb));
            let png = lcc_codecs::encode_png(&img, &lcc_codecs::EncodeMeta::default()).map_err(|e| e.to_string())?;
            let stem = Path::new(&p.raw).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| p.raw.clone());
            let out = dir.join(format!("{stem}.de00.png"));
            std::fs::write(&out, png).map_err(|e| format!("{}: {e}", out.display()))?;
        }
        let out = dir.join("verify.json");
        let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        std::fs::write(&out, text).map_err(|e| format!("{}: {e}", out.display()))?;
        eprintln!("calibrate-lightroom verify: wrote {} (and a ΔE00 map per photo)", out.display());
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    } else {
        print!("{}", verify::text(&report));
    }
    Ok(())
}

fn probe_cmd(args: &[String]) -> Result<(), String> {
    let (mut out, mut make, mut model, mut like, mut unique) = (None, None, None, None, None);
    let mut sweeps = true;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => out = Some(PathBuf::from(take(args, &mut i, "-o")?)),
            "--make" => make = Some(take(args, &mut i, "--make")?.to_string()),
            "--model" => model = Some(take(args, &mut i, "--model")?.to_string()),
            "--like" => like = Some(PathBuf::from(take(args, &mut i, "--like")?)),
            "--unique-model" => unique = Some(take(args, &mut i, "--unique-model")?.to_string()),
            "--base-only" => sweeps = false,
            a => return Err(format!("unknown argument `{a}`")),
        }
        i += 1;
    }
    let out = out.ok_or("probe needs -o DIR")?;
    if let Some(raw) = &like {
        let (mk, md) = lcc_colourfit::probe::camera_of(raw).map_err(|e| e.to_string())?;
        make = make.or(mk);
        model = model.or(md);
    }
    let (Some(make), Some(model)) = (make, model) else {
        return Err("probe needs the camera: --like ONE_OF_ITS_RAWS, or --make MAKE --model MODEL as its raws name it".into());
    };
    let m = lcc_colourfit::probe::write(&out, &make, &model, unique.as_deref(), sweeps).map_err(|e| e.to_string())?;
    eprintln!(
        "calibrate-lightroom probe: wrote {} probe DNG(s) for {} {} (UniqueCameraModel \"{}\") and {} to {}",
        m.files.len(),
        m.make,
        m.model,
        m.unique_camera_model,
        lcc_colourfit::probe::MANIFEST,
        out.display()
    );
    eprintln!(
        "next: import the folder into Lightroom, export every probe as a 16-bit ProPhoto TIFF keeping its file name, then run `calibrate-lightroom probe-check --probes {} --exports EXPORTS`",
        out.display()
    );
    Ok(())
}

fn probe_check_cmd(args: &[String]) -> Result<(), String> {
    let (mut probes, mut exports, mut out) = (None, None, None);
    let mut json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--probes" => probes = Some(PathBuf::from(take(args, &mut i, "--probes")?)),
            "--exports" => exports = Some(PathBuf::from(take(args, &mut i, "--exports")?)),
            "-o" | "--out" => out = Some(PathBuf::from(take(args, &mut i, "-o")?)),
            "--json" => json = true,
            a => return Err(format!("unknown argument `{a}`")),
        }
        i += 1;
    }
    let (Some(probes), Some(exports)) = (probes, exports) else {
        return Err("probe-check needs --probes DIR (from `probe`) and --exports DIR (Lightroom's exports of them)".into());
    };
    let report = lcc_colourfit::probe::check(&probes, &exports).map_err(|e| e.to_string())?;
    if let Some(path) = &out {
        let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
        eprintln!("calibrate-lightroom probe-check: wrote {}", path.display());
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?);
    } else {
        print!("{}", lcc_colourfit::probe::text(&report));
    }
    Ok(())
}

fn preset(args: &[String]) -> Result<(), String> {
    let mut out = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--out" => out = Some(take(args, &mut i, "-o")?.to_string()),
            a => return Err(format!("unknown argument `{a}`")),
        }
        i += 1;
    }
    let out = out.ok_or("preset needs -o FILE.xmp")?;
    let xmp = lcc_colourfit::preset::baseline_xmp();
    if xmp.is_empty() {
        return Err("the baseline preset is not available in this build yet".into());
    }
    std::fs::write(&out, xmp).map_err(|e| format!("{out}: {e}"))?;
    eprintln!("calibrate-lightroom: wrote {out}; import it in Lightroom's Develop module (Presets ▸ Import Presets)");
    Ok(())
}

/// Files below `dir` that look like raws (by content, else by extension), skipping hidden folders.
fn raws_in(path: &Path, out: &mut Vec<PathBuf>, depth: usize) {
    const EXT: &[&str] =
        &["arw", "cr2", "cr3", "crw", "dng", "nef", "nrw", "raf", "orf", "rw2", "rwl", "pef", "srw", "x3f", "3fr", "iiq", "erf", "mrw", "kdc", "dcr"];
    if path.is_file() {
        let by_ext = path.extension().and_then(|e| e.to_str()).is_some_and(|e| EXT.iter().any(|x| e.eq_ignore_ascii_case(x)));
        let by_content = || {
            use std::io::Read;
            let mut head = vec![0u8; 64 << 10];
            let n = std::fs::File::open(path).and_then(|mut f| f.read(&mut head)).unwrap_or(0);
            head.get(..n).and_then(lcc_codecs::sniff).is_some_and(|f| f.is_raw())
        };
        if by_ext || by_content() {
            out.push(path.to_path_buf());
        }
        return;
    }
    if depth > 16 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(path) else { return };
    for entry in entries.flatten() {
        if entry.file_name().to_string_lossy().starts_with(['.', '@']) {
            continue;
        }
        raws_in(&entry.path(), out, depth + 1);
    }
}

fn fit_cmd(args: &[String]) -> Result<(), String> {
    let (mut raw_dirs, mut export_dirs) = (Vec::new(), Vec::new());
    let mut out = PathBuf::from(".");
    let mut contributor = None;
    let mut dry_run = false;
    let mut lens = None;
    let mut editor = calibration::Editor::Lightroom;
    let mut value_divisions: Vec<usize> = Vec::new();
    let mut folds: Option<usize> = None;
    let mut allow_steep = false;
    let mut all_photos_profiles = false;
    let mut value_ending: Option<fit::ValueEnding> = None;
    let (mut low_sat_fade, mut bright_smooth): (Option<f64>, Option<f64>) = (None, None);
    let mut support_fade = false;
    let mut blue_pull = false;
    let (mut table_curvature, mut photo_cap) = (0.0f64, 0.0f64);
    // `--raws A B --exports C D`: plain arguments go to the list the last flag opened
    let mut list: Option<bool> = None; // Some(true) = raws, Some(false) = exports
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--raws" => list = Some(true),
            "--exports" => list = Some(false),
            "-o" | "--out" => out = take(args, &mut i, "-o")?.into(),
            "--contributor" => contributor = Some(take(args, &mut i, "--contributor")?.to_string()),
            "--dry-run" => dry_run = true,
            "--lens" => lens = Some(take(args, &mut i, "--lens")?.to_string()),
            "--value-divisions" => {
                let v = take(args, &mut i, "--value-divisions")?;
                for part in v.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                    let n: usize = part.parse().map_err(|_| format!("--value-divisions: `{part}` is not a number"))?;
                    if !(1..=fit::MAX_VALUE_DIVISIONS).contains(&n) {
                        return Err(format!("--value-divisions: {n} is not between 1 and {}", fit::MAX_VALUE_DIVISIONS));
                    }
                    value_divisions.push(n);
                }
            }
            "--folds" => {
                let v = take(args, &mut i, "--folds")?;
                folds = Some(v.parse().ok().filter(|n| *n >= 1).ok_or_else(|| format!("--folds: `{v}` is not a number of folds (1 or more)"))?);
            }
            "--allow-steep" => allow_steep = true,
            "--all-photos-profiles" => all_photos_profiles = true,
            "--value-support-fade" => support_fade = true,
            "--value-blue-pull" => blue_pull = true,
            "--table-smooth" => {
                let v = take(args, &mut i, "--table-smooth")?;
                table_curvature = v
                    .parse()
                    .ok()
                    .filter(|x: &f64| x.is_finite() && (0.0..=10000.0).contains(x))
                    .ok_or_else(|| format!("--table-smooth: `{v}` is not a number from 0 to 10000"))?;
            }
            "--photo-cap" => {
                let v = take(args, &mut i, "--photo-cap")?;
                photo_cap = v
                    .parse()
                    .ok()
                    .filter(|x: &f64| x.is_finite() && (0.0..1.0).contains(x))
                    .ok_or_else(|| format!("--photo-cap: `{v}` is not a share from 0 to below 1"))?;
            }
            "--value-low-sat-fade" | "--value-bright-smooth" => {
                let flag = args[i].clone();
                let v = take(args, &mut i, &flag)?;
                let x: f64 = v
                    .parse()
                    .ok()
                    .filter(|x: &f64| x.is_finite() && (0.0..=50.0).contains(x))
                    .ok_or_else(|| format!("{flag}: `{v}` is not a number from 0 to 50"))?;
                if flag == "--value-low-sat-fade" {
                    if x > 1.0 {
                        return Err(format!("{flag}: `{v}` is a saturation, from 0 to 1"));
                    }
                    low_sat_fade = Some(x);
                } else {
                    bright_smooth = Some(x);
                }
            }
            "--value-ending" => {
                let v = take(args, &mut i, "--value-ending")?;
                value_ending =
                    Some(fit::ValueEnding::parse(v).ok_or_else(|| format!("--value-ending: `{v}` is not one of pin, fade, hold, hold-sat"))?);
            }
            "--editor" => {
                let v = take(args, &mut i, "--editor")?;
                editor = calibration::Editor::parse(v).ok_or_else(|| {
                    let ids: Vec<&str> = calibration::Editor::ALL.iter().map(|e| e.id()).collect();
                    format!("--editor: `{v}` is not one of {}", ids.join(", "))
                })?;
            }
            a if a.starts_with('-') => return Err(format!("unknown option `{a}`")),
            a => match list {
                Some(true) => raw_dirs.push(PathBuf::from(a)),
                Some(false) => export_dirs.push(PathBuf::from(a)),
                None => return Err(format!("`{a}`: put folders after --raws or --exports")),
            },
        }
        i += 1;
    }
    if raw_dirs.is_empty() || export_dirs.is_empty() {
        return Err(format!("fit needs --raws DIR… and --exports DIR… (folders of {} exports, searched with their subfolders)", editor.name()));
    }
    value_divisions.retain(|n| *n > 1);
    value_divisions.sort_unstable();
    value_divisions.dedup();
    if folds.is_some() && value_divisions.is_empty() {
        return Err("--folds compares value divisions: give --value-divisions N[,N…] (above 1) with it".into());
    }
    if (allow_steep || all_photos_profiles || value_ending.is_some() || low_sat_fade.is_some() || bright_smooth.is_some() || support_fade || blue_pull)
        && value_divisions.is_empty()
    {
        return Err("--allow-steep, --all-photos-profiles and the --value-… options apply to value divisions: give --value-divisions N[,N…] (above 1) with them".into());
    }
    let mut raws = Vec::new();
    for d in &raw_dirs {
        raws_in(d, &mut raws, 0);
    }
    raws.sort();
    raws.dedup();
    if raws.is_empty() {
        return Err("no raw files found in the --raws folders".into());
    }
    let exports: Vec<ExportSet> = export_dirs.iter().map(|d| ExportSet { dir: d.clone() }).collect();
    eprintln!("calibrate-lightroom: {} raw file(s), {} export folder(s); pairing and sampling…", raws.len(), exports.len());
    let (photos, report) = dataset::collect(&raws, &exports, &CollectOptions::default()).map_err(|e| e.to_string())?;
    eprintln!("calibrate-lightroom: {} pair(s) usable", report.paired);
    for (file, why) in &report.skipped {
        eprintln!("  skipped {file}: {why}");
    }
    let ccts: Vec<f64> = photos.iter().filter_map(|p| p.white.lightroom_temp_tint.map(|(t, _)| t)).filter(|t| t.is_finite()).collect();
    if let (Some(lo), Some(hi)) = (ccts.iter().copied().reduce(f64::min), ccts.iter().copied().reduce(f64::max)) {
        eprintln!("calibrate-lightroom: white balance {lo:.0} K – {hi:.0} K over {} export(s)", ccts.len());
        if hi - lo < 1500.0 {
            eprintln!("  narrow white-balance range: add tungsten-lit (indoor) and daylight photos for a better fit");
        }
    }
    if editor != calibration::Editor::Lightroom && !ccts.is_empty() {
        eprintln!("  (the exports carry Lightroom white-balance values; fitting to {} ignores them)", editor.name());
    }
    let mut looks: Vec<String> = photos.iter().map(|p| fit::look_name_for(editor, &p.look)).collect();
    looks.sort_unstable();
    looks.dedup();
    eprintln!("calibrate-lightroom: fitting to {} exports; looks: {}", editor.name(), looks.join(", "));
    if dry_run {
        eprintln!("calibrate-lightroom: dry run, nothing fitted or written");
        return Ok(());
    }
    let mut cal = if value_divisions.is_empty() {
        eprintln!("calibrate-lightroom: fitting (this can take a few minutes)…");
        fit_once(&photos, &FitOptions { contributor, editor, rounds: 0, table_curvature, photo_cap, ..Default::default() })?
    } else {
        let folds = folds.unwrap_or(1).min(dataset::folds(CollectOptions::default().hold_out));
        let opts = FitOptions {
            contributor,
            editor,
            rounds: 0,
            value_ending: value_ending.unwrap_or_default(),
            value_low_sat_fade: low_sat_fade.unwrap_or(0.0),
            value_bright_smooth: bright_smooth.unwrap_or(0.0),
            value_support_fade: support_fade,
            value_blue_pull: blue_pull,
            table_curvature,
            photo_cap,
            ..Default::default()
        };
        value_division_search(photos, &opts, &value_divisions, folds, allow_steep, all_photos_profiles, &out)?
    };
    // the lens the calibration is for (`--lens`), set before validating as a plain fit always did
    cal.lens = lens;
    cal.validate().map_err(|e| format!("the fit produced an invalid calibration: {e}"))?;
    let (json_path, dcps) = write_calibration(&cal, &out)?;
    print_contents(&cal);
    println!();
    println!("Send this file: {}", json_path.display());
    println!("  It holds only the numbers listed above: no images, file names, dates, serial numbers or locations (a lens model only with --lens).");
    if lcc_brand::CALIBRATION_DISCORD.is_empty() {
        println!("  Open an issue on {} and attach it (licence {}).", lcc_brand::REPOSITORY, calibration::LICENCE);
    } else {
        let thread =
            if lcc_brand::CALIBRATION_THREAD.is_empty() { String::new() } else { format!(" in the {} thread", lcc_brand::CALIBRATION_THREAD) };
        println!("  Send it on Discord{thread}: {} (licence {}).", lcc_brand::CALIBRATION_DISCORD, calibration::LICENCE);
    }
    if !dcps.is_empty() {
        println!("For your own testing only (do not send): Camera ▸ Import DCP… in the app, or `camera.importDcp`:");
        for p in &dcps {
            println!("  {}", p.display());
        }
    }
    Ok(())
}

/// One fit, stamped with this tool's version and validated.
fn fit_once(photos: &[dataset::PhotoSamples], opts: &FitOptions) -> Result<Calibration, String> {
    let mut cal = fit::fit(photos, opts).map_err(|e| e.to_string())?;
    cal.tool_version = env!("CARGO_PKG_VERSION").to_string();
    cal.validate().map_err(|e| format!("the fit produced an invalid calibration: {e}"))?;
    Ok(cal)
}

/// Write the calibration JSON and one `.dcp` per look into `out`: (JSON path, `.dcp` paths).
fn write_calibration(cal: &Calibration, out: &Path) -> Result<(PathBuf, Vec<PathBuf>), String> {
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let json_path = out.join(cal.file_name());
    std::fs::write(&json_path, cal.to_json().map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", json_path.display()))?;
    let mut dcps = Vec::new();
    for look in &cal.looks {
        let dcp = cal.look_dcp(look);
        let bytes = write_dcp(&dcp.model, &dcp.name, &dcp.copyright, &dcp.color)?;
        let stem = json_path.file_stem().and_then(|s| s.to_str()).unwrap_or("calibration");
        // the id without its `cal:` prefix: `:` is not allowed in Windows file names
        let id = calibration::look_id(&look.name);
        let slug = id.strip_prefix(calibration::LOOK_PREFIX).unwrap_or(&id);
        let path = out.join(format!("{stem}.{slug}.dcp"));
        std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        dcps.push(path);
    }
    Ok((json_path, dcps))
}

/// A fit's held-out score: the base look's ΔE00 median and p95, and how steeply its hue/saturation map changes per
/// stop of brightness at [`STOP_QUANTILE`] of its cells (the measure [`STOP_LIMIT_WARM`] and [`STOP_LIMIT_BLUE`] apply
/// to).
#[derive(Clone, Copy, Debug, PartialEq)]
struct FoldScore {
    median: f64,
    p95: f64,
    steep: fit::StopSteepness,
    /// How far its near-white tints are from the same fold's single-step fit ([`fit::near_white_shift`] at
    /// [`STOP_QUANTILE`]; 0 for the single-step fit itself).
    near_white: f64,
    /// The worst photo's noise excess of its maps ([`fit::noise_excess`], over every photo of the fold).
    noise: f64,
}

fn score_of(cal: &Calibration, single_step: Option<&Calibration>, photos: &[dataset::PhotoSamples]) -> Option<FoldScore> {
    let l = cal.evidence.looks.iter().filter(|l| l.held_out > 0).max_by_key(|l| l.photos)?;
    Some(FoldScore {
        median: f64::from(l.after.median),
        p95: f64::from(l.after.p95),
        steep: fit::value_steepness_per_stop(cal, STOP_QUANTILE),
        near_white: single_step.map_or(0.0, |b| fit::near_white_shift(cal, b, STOP_QUANTILE)),
        noise: fit::noise_excess(cal, photos).0,
    })
}

// The noise excess ([`fit::noise_excess`]) is reported, not part of the rule: on the Z 8 data, measured on the training
// samples, it names the photo a mottled fit is worst on (a haze photo whose own colours sit in a cell that jumps away
// from its neighbours), but it also scores fits that render that photo cleanly nearly as high (a misty sea photo at
// 1.65 against 2.10 for the mottled fit), so no limit separates the two yet. Full-size renders of the named photos
// (flat-patch noise against the built-in profile) are the check that does.

/// A value-division candidate loses a single fold by a clear margin when it is worse than the single-step map there by
/// more than this much ΔE00 (median, p95).
const FOLD_MARGIN: (f64, f64) = (0.10, 0.25);
/// The steepness rule's measure: this quantile of the map's per-stop changes (cells pooled over both illuminant maps and
/// every value step but the one from black, [`fit::value_steepness_per_stop`]) must stay within the limits below in
/// every fold (the worst fold counts, not the mean).
const STOP_QUANTILE: f64 = 0.99;
/// Steepest change per stop of brightness the oranges/yellows may have (hue °/stop, |ln saturation ratio|/stop):
/// Capture One's own 99th percentile, 4.6° and 0.28, measured from exports of the same photos at −1, 0 and +1 EV, plus
/// 25 %. The reference comes from 5 scenes, and the 25 % tolerance is a judgement call.
const STOP_LIMIT_WARM: (f64, f64) = (5.75, 0.35);
/// As [`STOP_LIMIT_WARM`] for the blues: Capture One's 3.0° and 0.24 plus 25 %, measured on a single dusk sky.
const STOP_LIMIT_BLUE: (f64, f64) = (3.75, 0.30);
/// Furthest a candidate may move near-white tints from the same fold's single-step fit ([`fit::near_white_shift`] at
/// [`STOP_QUANTILE`], ΔE*ab of a*b*), in every fold. Bright near-neutrals (clouds, highlights) have almost no training
/// samples and no Capture One reference in the exposure series, so the brightness-dependent part should leave them
/// about where the single-step map puts them. The limit is a judgement call: about a just-noticeable tint difference.
const NEAR_WHITE_LIMIT: f64 = 2.0;

fn within_stop_limits(s: &fit::StopSteepness) -> bool {
    s.warm.0 <= STOP_LIMIT_WARM.0 && s.warm.1 <= STOP_LIMIT_WARM.1 && s.blue.0 <= STOP_LIMIT_BLUE.0 && s.blue.1 <= STOP_LIMIT_BLUE.1
}

fn stop_text(s: &fit::StopSteepness) -> String {
    format!(
        "warm {:.2}°/{:.3}, blue {:.2}°/{:.3} per stop (limits {}°/{}, {}°/{})",
        s.warm.0, s.warm.1, s.blue.0, s.blue.1, STOP_LIMIT_WARM.0, STOP_LIMIT_WARM.1, STOP_LIMIT_BLUE.0, STOP_LIMIT_BLUE.1
    )
}
/// How much lower a larger candidate's mean held-out p95 must be to be chosen over a smaller one that also passes.
const CLEARLY_BETTER_P95: f64 = 0.15;

fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let (s, n) = v.fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 { f64::NAN } else { s / n as f64 }
}

/// Choose the value divisions from per-fold scores (`scores[k][fold]`, `scores[0]` the single-step fit): a candidate
/// must have every fold, beat the single-step fit on the mean median and the mean p95, not lose any fold by
/// [`FOLD_MARGIN`], and stay within [`STOP_LIMIT_WARM`] and [`STOP_LIMIT_BLUE`] in every fold; of those that pass the smallest is chosen unless
/// a larger one's mean p95 is [`CLEARLY_BETTER_P95`] lower. Returns the chosen divisions (1 when none passes), one
/// line per candidate saying why, and the candidates that fail the steepness check and nothing else.
fn choose_value_divisions(ns: &[usize], scores: &[Vec<Option<FoldScore>>]) -> (usize, Vec<String>, Vec<usize>) {
    let mut why = Vec::new();
    let mut steep_only = Vec::new();
    let Some(base) = scores.first() else { return (1, why, steep_only) };
    if base.iter().any(Option::is_none) {
        why.push("the single-step fit failed in a fold, so nothing can be compared with it".into());
        return (1, why, steep_only);
    }
    let base: Vec<FoldScore> = base.iter().flatten().copied().collect();
    let (bm, bp) = (mean(base.iter().map(|s| s.median)), mean(base.iter().map(|s| s.p95)));
    let mut passing: Vec<(usize, f64)> = Vec::new();
    for (n, sc) in ns.iter().zip(scores).skip(1) {
        if sc.iter().any(Option::is_none) || sc.len() != base.len() {
            why.push(format!("N={n}: a fold failed to fit"));
            continue;
        }
        let sc: Vec<FoldScore> = sc.iter().flatten().copied().collect();
        let (m, p) = (mean(sc.iter().map(|s| s.median)), mean(sc.iter().map(|s| s.p95)));
        let mut fails = Vec::new();
        let mut steep = false;
        let mut steep_folds = Vec::new();
        if m.partial_cmp(&bm) != Some(std::cmp::Ordering::Less) {
            fails.push(format!("mean median {m:.3} not below {bm:.3}"));
        }
        if p.partial_cmp(&bp) != Some(std::cmp::Ordering::Less) {
            fails.push(format!("mean p95 {p:.3} not below {bp:.3}"));
        }
        for (f, (s, b)) in sc.iter().zip(&base).enumerate() {
            if s.median > b.median + FOLD_MARGIN.0 || s.p95 > b.p95 + FOLD_MARGIN.1 {
                fails.push(format!("fold {f} lost clearly (median {:.2} vs {:.2}, p95 {:.2} vs {:.2})", s.median, b.median, s.p95, b.p95));
            }
            if !within_stop_limits(&s.steep) {
                steep = true;
                steep_folds.push(format!("fold {f}: {}", stop_text(&s.steep)));
            }
            if s.near_white > NEAR_WHITE_LIMIT {
                steep = true;
                steep_folds.push(format!("fold {f}: near-white tints {:.2} ΔE*ab from the single-step fit (limit {NEAR_WHITE_LIMIT})", s.near_white));
            }
        }
        let accurate = fails.is_empty();
        if steep {
            fails.push(format!("too steep per stop (99th percentile) in {}", steep_folds.join("; ")));
            if accurate {
                steep_only.push(*n);
            }
        }
        if fails.is_empty() {
            why.push(format!("N={n}: passes (mean median {m:.3} vs {bm:.3}, mean p95 {p:.3} vs {bp:.3})"));
            passing.push((*n, p));
        } else {
            why.push(format!("N={n}: fails: {}", fails.join("; ")));
        }
    }
    let mut chosen: Option<(usize, f64)> = None;
    for (n, p) in passing {
        chosen = match chosen {
            Some((cn, cp)) if p > cp - CLEARLY_BETTER_P95 => Some((cn, cp)),
            _ => Some((n, p)),
        };
    }
    (chosen.map_or(1, |c| c.0), why, steep_only)
}

/// `--value-divisions N[,N…] [--folds K]`: fit the single-step map and each candidate on every fold, print the
/// per-fold held-out numbers, choose by [`choose_value_divisions`] and return the chosen fit of fold 0 (the default
/// hold-out). Every fit is also written to `out/value-divisions/fold-K/N…` for inspection.
fn value_division_search(
    mut photos: Vec<dataset::PhotoSamples>,
    opts: &FitOptions,
    candidates: &[usize],
    folds: usize,
    allow_steep: bool,
    all_photos_profiles: bool,
    out: &Path,
) -> Result<Calibration, String> {
    let share = CollectOptions::default().hold_out;
    let mut ns = vec![1];
    ns.extend_from_slice(candidates);
    let mut scores: Vec<Vec<Option<FoldScore>>> = vec![Vec::new(); ns.len()];
    let mut fold0: Vec<Option<Calibration>> = vec![None; ns.len()];
    for fold in 0..folds.max(1) {
        dataset::hold_out_fold(&mut photos, share, fold);
        // this fold's single-step fit, which the candidates' near-white tints are compared with
        let mut single_step: Option<Calibration> = None;
        for (k, &n) in ns.iter().enumerate() {
            eprintln!("calibrate-lightroom: fold {}/{} N={n}: fitting…", fold + 1, folds.max(1));
            // every fit is scored against the others, even one that wouldn't beat the app's current colour on this
            // fold; the gate is applied to the one written at the end
            let o = FitOptions { value_divisions: n, report_warm: true, keep_ungated: true, ..opts.clone() };
            let got = match fit_once(&photos, &o) {
                Ok(cal) => {
                    let dir = out.join("value-divisions").join(format!("fold-{fold}")).join(format!("N{n}"));
                    write_calibration(&cal, &dir)?;
                    let s = score_of(&cal, if n == 1 { None } else { single_step.as_ref() }, &photos);
                    if n == 1 {
                        single_step = Some(cal.clone());
                    }
                    if let Some(s) = s {
                        let max = fit::value_steepness_per_stop(&cal, 1.0);
                        eprintln!(
                            "calibrate-lightroom: fold {fold} N={n}: held-out ΔE00 median {:.2}, p95 {:.2}{}; steepness 99th percentile {}{}; \
                             largest warm {:.2}°/{:.3}, blue {:.2}°/{:.3} per stop; near-white tints {:.2} ΔE*ab from the single-step fit{}; \
                             noise excess {:.2} on the worst photo (reported, not a rule)",
                            s.median,
                            s.p95,
                            if fit::beats_current_colour(&cal) { "" } else { " (does not beat the app's current colour)" },
                            stop_text(&s.steep),
                            if within_stop_limits(&s.steep) { "" } else { " TOO STEEP" },
                            max.warm.0,
                            max.warm.1,
                            max.blue.0,
                            max.blue.1,
                            s.near_white,
                            if s.near_white > NEAR_WHITE_LIMIT { " TOO FAR" } else { "" },
                            s.noise
                        );
                        // the photos it is worst on (local console only, like the skipped-file list)
                        let worst: Vec<String> = fit::noise_excess_per_photo(&cal, &photos)
                            .iter()
                            .take(3)
                            .map(|(i, e)| format!("{} {e:.2}", photos[*i].raw_path.file_name().map_or_else(String::new, |f| f.to_string_lossy().into_owned())))
                            .collect();
                        eprintln!("calibrate-lightroom: fold {fold} N={n}: noise excess worst photos: {}", worst.join(", "));
                    }
                    if fold == 0
                        && let Some(slot) = fold0.get_mut(k)
                    {
                        *slot = Some(cal);
                    }
                    s
                }
                Err(e) => {
                    eprintln!("calibrate-lightroom: fold {fold} N={n}: no fit: {e}");
                    None
                }
            };
            if let Some(v) = scores.get_mut(k) {
                v.push(got);
            }
        }
    }
    println!("Held-out ΔE00 per fold (median / p95):");
    for (n, sc) in ns.iter().zip(&scores) {
        let cells: Vec<String> = sc.iter().map(|s| s.map_or("   failed   ".into(), |s| format!("{:5.2} / {:5.2}", s.median, s.p95))).collect();
        let ok: Vec<FoldScore> = sc.iter().flatten().copied().collect();
        println!("  N={n:<2} {}  | mean {:.2} / {:.2}", cells.join("  "), mean(ok.iter().map(|s| s.median)), mean(ok.iter().map(|s| s.p95)));
    }
    if folds <= 1 {
        println!("  (one held-out split only: pass --folds 5 for a decision that doesn't rest on one choice of photos)");
    }
    let (chosen, why, steep_only) = choose_value_divisions(&ns, &scores);
    for w in &why {
        println!("  {w}");
    }
    if allow_steep {
        write_steep(&photos, opts, &steep_only, &out.join("steep"))?;
    }
    if all_photos_profiles {
        for &n in candidates {
            let passes = why.iter().any(|w| w.starts_with(&format!("N={n}: passes")));
            let rule = if passes { "passes-rule" } else { "fails-rule" };
            let words = if passes { "passes the rule" } else { "fails the rule" };
            let label = variant_label(opts, n);
            let (path, name, p99) = write_all_photos_fit(
                &photos,
                opts,
                n,
                &out.join("all-photos"),
                &format!("{label}.{rule}.all-photos-not-held-out-checked"),
                &format!("{label}, {words}, all photos, not checked on held-out photos"),
            )?;
            println!(
                "ALL-PHOTOS TEST PROFILE ({words}; not checked on held-out photos): {}\n  profile name \"{name}\"; steepness 99th percentile {}",
                path.display(),
                stop_text(&p99)
            );
        }
    }
    let k = ns.iter().position(|n| *n == chosen).unwrap_or(0);
    let cal = fold0.get_mut(k).and_then(Option::take).or_else(|| fold0.first_mut().and_then(Option::take));
    let cal = cal.ok_or_else(|| "the single-step fit of the default hold-out failed".to_string())?;
    // the written fit passes the gate a plain fit applies
    if !fit::beats_current_colour(&cal) {
        return Err(format!(
            "the {} fit of the default hold-out does not beat the app's current colour on its held-out photos: nothing written \
             (the per-fold fits are in {})",
            if chosen == 1 { "single-step".to_string() } else { format!("N={chosen}") },
            out.join("value-divisions").display()
        ));
    }
    if chosen == 1 {
        println!("Value divisions: none of {candidates:?} passed; writing the single-step fit (fold 0).");
    } else {
        println!("Value divisions: {chosen} chosen; writing its fit of the default hold-out (fold 0).");
    }
    Ok(cal)
}

/// A candidate's label in test-profile names: `N6-hold`, plus the experimental shape options when set
/// (`-lowsat0.25`, `-bright8`).
fn variant_label(opts: &FitOptions, n: usize) -> String {
    let mut s = format!("N{n}-{}", opts.value_ending.id());
    if opts.value_low_sat_fade > 0.0 {
        s += &format!("-lowsat{}", opts.value_low_sat_fade);
    }
    if opts.value_bright_smooth > 0.0 {
        s += &format!("-bright{}", opts.value_bright_smooth);
    }
    if opts.value_support_fade {
        s += "-support";
    }
    if opts.value_blue_pull {
        s += "-bluepull";
    }
    if opts.table_curvature > 0.0 {
        s += &format!("-smooth{}", opts.table_curvature);
    }
    if opts.photo_cap > 0.0 {
        s += &format!("-cap{}", opts.photo_cap);
    }
    s
}

/// `--all-photos-profiles`: candidate `n` fitted on every photo, none held out, written as `.dcp` files only into `dir`
/// with `file_tag` in the file name and `(name_tag)` after the profile name, so it imports next to the profiles already
/// there. Returns the (last) path, its profile name and its steepness.
fn write_all_photos_fit(
    photos: &[dataset::PhotoSamples],
    opts: &FitOptions,
    n: usize,
    dir: &Path,
    file_tag: &str,
    name_tag: &str,
) -> Result<(PathBuf, String, fit::StopSteepness), String> {
    eprintln!("calibrate-lightroom: --all-photos-profiles: {name_tag}: fitting…");
    let o = FitOptions { value_divisions: n, keep_ungated: true, all_photos: true, ..opts.clone() };
    let cal = fit_once(photos, &o)?;
    let p99 = fit::value_steepness_per_stop(&cal, STOP_QUANTILE);
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut last = (PathBuf::new(), String::new());
    for look in &cal.looks {
        let mut dcp = cal.look_dcp(look);
        dcp.name = format!("{} ({name_tag})", dcp.name);
        let bytes = write_dcp(&dcp.model, &dcp.name, &dcp.copyright, &dcp.color)?;
        let id = calibration::look_id(&look.name);
        let slug = id.strip_prefix(calibration::LOOK_PREFIX).unwrap_or(&id);
        let stem = Path::new(&cal.file_name()).file_stem().and_then(|s| s.to_str()).unwrap_or("calibration").to_string();
        let path = dir.join(format!("{stem}.{slug}.{file_tag}.dcp"));
        std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        last = (path, dcp.name);
    }
    Ok((last.0, last.1, p99))
}

/// `--allow-steep`: each candidate in `ns` (those that fail the steepness check and nothing else) fitted on every
/// photo, none held out, and written as `.dcp` files only into `dir`, labelled in the file name and in the profile name
/// (`… (N6 steep, untested)`), so they import next to the profiles already there instead of replacing one. No
/// calibration JSON is written for them: they are not checked on held-out photos and are not for sending.
fn write_steep(photos: &[dataset::PhotoSamples], opts: &FitOptions, ns: &[usize], dir: &Path) -> Result<(), String> {
    if ns.is_empty() {
        println!("--allow-steep: no candidate fails only the steepness check; no steep profiles written.");
        return Ok(());
    }
    let ending = opts.value_ending;
    for &n in ns {
        eprintln!("calibrate-lightroom: --allow-steep: N={n} ({} ending), fitting on all photos…", ending.id());
        let o = FitOptions { value_divisions: n, keep_ungated: true, all_photos: true, ..opts.clone() };
        let cal = fit_once(photos, &o)?;
        let p99 = fit::value_steepness_per_stop(&cal, STOP_QUANTILE);
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let label = variant_label(opts, n);
        for look in &cal.looks {
            let mut dcp = cal.look_dcp(look);
            dcp.name = format!("{} ({label} steep, untested)", dcp.name);
            let bytes = write_dcp(&dcp.model, &dcp.name, &dcp.copyright, &dcp.color)?;
            let id = calibration::look_id(&look.name);
            let slug = id.strip_prefix(calibration::LOOK_PREFIX).unwrap_or(&id);
            let stem = Path::new(&cal.file_name()).file_stem().and_then(|s| s.to_str()).unwrap_or("calibration").to_string();
            let path = dir.join(format!("{stem}.{slug}.STEEP-{label}-all-photos.dcp"));
            std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            println!(
                "STEEP PROFILE (fails the steepness check, fitted on all photos, not checked on held-out photos): {}\n  \
                 profile name \"{}\"; steepness 99th percentile {}",
                path.display(),
                dcp.name,
                stop_text(&p99)
            );
        }
    }
    Ok(())
}

/// Everything a calibration file holds, in words.
fn print_contents(c: &Calibration) {
    println!("{} {}: calibration matched to {}", c.make, c.model, c.editor.name());
    if let Some(l) = &c.lens {
        println!("  lens: {l} (only photos taken with this lens use it)");
    }
    if !c.aliases.is_empty() {
        println!("  also sold as: {}", c.aliases.join(", "));
    }
    println!("  contributor: {}", c.contributor.as_deref().unwrap_or("(none; pass --contributor NAME for credit)"));
    println!("  tool version {}, licence {}", c.tool_version, c.licence);
    println!(
        "  camera colour model: illuminants {:?}, two colour and two forward matrices (3×3){}",
        c.camera.illuminants,
        match &c.camera.hue_sat_map {
            Some([t, _]) if t.val_divisions > 1 => format!(", two hue/saturation maps with {} brightness steps", t.val_divisions),
            Some(_) => ", two hue/saturation maps".to_string(),
            None => String::new(),
        }
    );
    for l in &c.looks {
        println!(
            "  look \"{}\": exposure offset {:+.2} EV, {}-point tone curve{}",
            l.name,
            l.baseline_exposure_offset,
            l.tone_curve.len(),
            if l.look_table.is_some() { ", look table" } else { "" }
        );
    }
    let e = &c.evidence;
    println!("  evidence: {} photos, {} colour samples, hue coverage {:.0} %", e.photos, e.samples, e.hue_coverage * 100.0);
    if !e.cct.is_empty() {
        let bins: Vec<String> = e.cct.iter().map(|b| format!("≥{} K: {}", b.from_k, b.photos)).collect();
        println!("  white-balance ranges (photos): {}", bins.join(", "));
    }
    for l in &e.looks {
        println!(
            "  held-out ΔE00 \"{}\" ({} photos, {} held out): median {:.2} → {:.2}, p95 {:.2} → {:.2}",
            l.name, l.photos, l.held_out, l.before.median, l.after.median, l.before.p95, l.after.p95
        );
    }
}

fn list() -> Result<(), String> {
    let all = calibration::embedded();
    if all.is_empty() {
        println!("no calibrations are built in yet");
        return Ok(());
    }
    for c in all {
        let looks: Vec<String> =
            c.looks.iter().map(|l| format!("{} ({}{})", l.name, calibration::LOOK_PREFIX, calibration::look_id(&l.name))).collect();
        println!("{} {} ({}): {} photos; looks: {}", c.make, c.model, c.editor.name(), c.evidence.photos, looks.join(", "));
        for l in &c.evidence.looks {
            println!("  ΔE00 {}: median {:.2} → {:.2}, p95 {:.2} → {:.2}", l.name, l.before.median, l.after.median, l.before.p95, l.after.p95);
        }
    }
    Ok(())
}

fn add(args: &[String]) -> Result<(), String> {
    let mut file = None;
    let mut repo = PathBuf::from(".");
    let mut contributor = None;
    let mut lens = None;
    let mut replace = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--repo" => repo = take(args, &mut i, "--repo")?.into(),
            "--contributor" => contributor = Some(take(args, &mut i, "--contributor")?.to_string()),
            "--lens" => lens = Some(take(args, &mut i, "--lens")?.to_string()),
            "--replace" => replace = true,
            a if a.starts_with('-') => return Err(format!("unknown option `{a}`")),
            a if file.is_none() => file = Some(a.to_string()),
            a => return Err(format!("unexpected argument `{a}`")),
        }
        i += 1;
    }
    let file = file.ok_or("add needs the contributed calibration FILE")?;
    let meta = std::fs::metadata(&file).map_err(|e| format!("{file}: {e}"))?;
    if meta.len() > calibration::MAX_FILE as u64 {
        return Err(format!("{file}: {} bytes is too large for a calibration", meta.len()));
    }
    let text = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
    let mut cal = Calibration::parse(&text).map_err(|e| format!("{file}: {e}"))?;
    if contributor.is_some() || lens.is_some() {
        if let Some(c) = contributor {
            cal.contributor = Some(c);
        }
        if let Some(l) = lens {
            cal.lens = Some(l);
        }
        cal.validate().map_err(|e| format!("{file}: {e}"))?;
    }
    let attribution = repo.join("assets/ATTRIBUTION.md");
    let table =
        std::fs::read_to_string(&attribution).map_err(|e| format!("{}: {e} (pass --repo DIR, the repository root)", attribution.display()))?;
    let dir = repo.join(DATA_DIR);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let name = cal.file_name();
    let dest = dir.join(&name);
    if !replace {
        let taken = calibration_files(&dir).into_iter().find(|(path, existing)| path == &dest || existing.same_slot(&cal));
        if let Some((path, _)) = taken {
            let lens = cal.lens.as_deref().map(|l| format!(", lens {l}")).unwrap_or_default();
            return Err(format!(
                "{} already holds a {} calibration for {} {}{lens}; pass --replace to replace it, or --lens MODEL if this fit is for another lens",
                path.display(),
                cal.editor.name(),
                cal.make,
                cal.model
            ));
        }
    }
    std::fs::write(&dest, cal.to_json().map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", dest.display()))?;
    let today = chrono::DateTime::<chrono::Utc>::from(std::time::SystemTime::now()).date_naive().format("%Y-%m-%d").to_string();
    let updated = with_attribution_row(&table, &format!("{DATA_DIR}/{name}"), &cal, &today);
    std::fs::write(&attribution, updated).map_err(|e| format!("{}: {e}", attribution.display()))?;
    println!("added {}", dest.display());
    println!("updated {}", attribution.display());
    print_contents(&cal);
    Ok(())
}

/// The calibrations already in `dir` (files that don't read as one are left out: they hold no slot).
fn calibration_files(dir: &Path) -> Vec<(PathBuf, Calibration)> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("json")))
        .filter(|p| std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.len() <= calibration::MAX_FILE as u64))
        .filter_map(|p| {
            let c = Calibration::parse(&std::fs::read_to_string(&p).ok()?).ok()?;
            Some((p, c))
        })
        .collect()
}

/// The ATTRIBUTION table with `path`'s row added after the last calibration or camera-profile row, or replacing
/// the row it already has.
fn with_attribution_row(table: &str, path: &str, c: &Calibration, today: &str) -> String {
    let looks: Vec<&str> = c.looks.iter().map(|l| l.name.as_str()).collect();
    let by = c.contributor.as_deref().map(|n| format!(" by {}", n.replace('|', "/"))).unwrap_or_default();
    let row = format!(
        "| `{path}` | {make} {model} camera calibration matched to {editor} (colour model + looks: {looks}), built into the app | {who} | original work: black-box fit by `<cli> calibrate-lightroom` to {n} photos and their {editor} exports{by}; aggregate statistics only, no image content; no profile data of the editor's | {lic} | {today} | none |",
        editor = c.editor.name(),
        make = c.make.replace('|', "/"),
        model = c.model.replace('|', "/"),
        looks = looks.join(", ").replace('|', "/"),
        who = c.contributor.as_deref().map(|n| n.replace('|', "/")).unwrap_or_else(|| "contributors".into()),
        n = c.evidence.photos,
        lic = c.licence,
    );
    let key = format!("| `{path}` |");
    let mut lines: Vec<String> = table.lines().map(str::to_string).collect();
    if let Some(l) = lines.iter_mut().find(|l| l.starts_with(&key)) {
        *l = row;
    } else {
        let anchor = lines
            .iter()
            .rposition(|l| l.starts_with(&format!("| `{DATA_DIR}/")))
            .or_else(|| lines.iter().rposition(|l| l.starts_with("| `assets/camera-profiles/")))
            .or_else(|| lines.iter().rposition(|l| l.starts_with("|---")));
        match anchor {
            Some(at) => lines.insert(at + 1, row),
            None => lines.push(row),
        }
    }
    let mut s = lines.join("\n");
    s.push('\n');
    s
}

/// A little-endian DNG camera profile (`.dcp`) of `color`'s profile tags, readable by `lcc_raw::dcp::parse`.
pub fn write_dcp(model: &str, name: &str, copyright: &str, color: &ColorData) -> Result<Vec<u8>, String> {
    lcc_raw::dcp::write(&lcc_raw::dcp::Dcp { model: model.into(), name: name.into(), copyright: copyright.into(), color: color.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lcc_raw::Mat3;
    use lcc_raw::profile::ToneCurve;

    fn sc(median: f64, p95: f64) -> Option<FoldScore> {
        Some(FoldScore { median, p95, steep: fit::StopSteepness { warm: (4.0, 0.2), blue: (2.0, 0.2) }, near_white: 0.5, noise: 0.3 })
    }

    #[test]
    fn value_divisions_are_chosen_across_folds() {
        let base = vec![sc(2.0, 5.0), sc(2.1, 5.2), sc(1.9, 4.8)];
        // N=4 better everywhere, N=6 a little better still on p95 but not clearly: the smaller one wins
        let four = vec![sc(1.9, 4.7), sc(2.0, 4.9), sc(1.8, 4.6)];
        let six = vec![sc(1.9, 4.6), sc(2.0, 4.8), sc(1.8, 4.6)];
        assert_eq!(choose_value_divisions(&[1, 4, 6], &[base.clone(), four.clone(), six]).0, 4);
        // clearly better on p95: the larger one wins
        let six = vec![sc(1.85, 4.4), sc(1.95, 4.6), sc(1.75, 4.4)];
        assert_eq!(choose_value_divisions(&[1, 4, 6], &[base.clone(), four.clone(), six]).0, 6);
        // better on average but losing one fold by a clear margin: refused
        let lumpy = vec![sc(1.5, 4.0), sc(2.3, 5.6), sc(1.5, 4.0)];
        let (n, why, steep_only) = choose_value_divisions(&[1, 4], &[base.clone(), lumpy.clone()]);
        assert_eq!(n, 1);
        assert!(why.iter().any(|w| w.contains("fold 1 lost clearly")), "{why:?}");
        assert!(steep_only.is_empty());
        // too steep along value, or a failed fold: refused; only the steep one counts as failing on steepness alone
        let steep = vec![
            Some(FoldScore { steep: fit::StopSteepness { warm: (4.0, 0.36), blue: (2.0, 0.2) }, ..sc(1.5, 4.0).unwrap() }),
            sc(1.5, 4.0),
            sc(1.5, 4.0),
        ];
        let (n, _, steep_only) = choose_value_divisions(&[1, 4], &[base.clone(), steep.clone()]);
        assert_eq!((n, steep_only), (1, vec![4]));
        let steep_lumpy: Vec<Option<FoldScore>> =
            lumpy.iter().map(|s| s.map(|s| FoldScore { steep: fit::StopSteepness { warm: (4.0, 0.2), blue: (3.8, 0.2) }, ..s })).collect();
        assert!(choose_value_divisions(&[1, 4, 6], &[base.clone(), steep, steep_lumpy]).2 == vec![4]);
        assert_eq!(choose_value_divisions(&[1, 4], &[base.clone(), vec![sc(1.0, 3.0), None, sc(1.0, 3.0)]]).0, 1);
        // better on p95 but not on the median: refused
        assert_eq!(choose_value_divisions(&[1, 4], &[base.clone(), vec![sc(2.05, 4.0), sc(2.15, 4.2), sc(1.95, 3.8)]]).0, 1);
        // the single-step fit failing a fold leaves nothing to compare with
        assert_eq!(choose_value_divisions(&[1, 4], &[vec![None, sc(2.0, 5.0)], vec![sc(1.0, 3.0), sc(1.0, 3.0)]]).0, 1);
        assert_eq!(choose_value_divisions(&[], &[]).0, 1);
    }

    #[test]
    fn written_profiles_parse_back() {
        let mut color = ColorData { illuminant: [17, 21], ..ColorData::default() };
        let fm = Mat3([[0.70, 0.18, 0.08], [0.28, 0.80, -0.08], [0.02, -0.15, 0.95]]);
        let cm = Mat3([[0.9, -0.3, -0.05], [-0.5, 1.4, 0.1], [-0.1, 0.3, 0.6]]);
        color.color_matrix = [Some(cm), Some(cm)];
        color.forward_matrix = [Some(fm), Some(fm)];
        color.baseline_exposure = 0.35;
        let table =
            lcc_raw::profile::HsvTable { hue_divisions: 6, sat_divisions: 2, val_divisions: 1, data: vec![[2.0, 1.1, 0.9]; 12], srgb_value: false };
        color.profile.hue_sat_map = [Some(table.clone()), Some(table.clone())];
        color.profile.look_table = Some(table);
        color.profile.tone_curve = Some(ToneCurve { points: vec![[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]] });
        let bytes = write_dcp("SONY ILCE-7M4", "Standard (matched)", "A Contributor", &color).unwrap();
        let d = lcc_raw::dcp::parse(&bytes).unwrap();
        assert_eq!(d.model, "SONY ILCE-7M4");
        assert_eq!(d.name, "Standard (matched)");
        assert_eq!(d.copyright, "A Contributor");
        assert_eq!(d.color.illuminant, [17, 21]);
        let got = d.color.forward_matrix[1].unwrap();
        assert!((got.0[1][1] - 0.80).abs() < 1e-3);
        assert!((d.color.baseline_exposure - 0.35).abs() < 1e-3);
        assert!(d.color.profile.hue_sat_map[1].is_some());
        assert_eq!(d.color.profile.look_table.as_ref().map(|t| t.data.len()), Some(12));
        assert_eq!(d.color.profile.tone_curve.as_ref().map(|c| c.points.len()), Some(3));
        // a map with value divisions keeps them and its sRGB-encoded value axis
        let deep = lcc_raw::profile::HsvTable {
            hue_divisions: 6,
            sat_divisions: 2,
            val_divisions: 4,
            data: (0..48).map(|i| [i as f32 * 0.1, 1.0 + i as f32 * 0.01, 1.0]).collect(),
            srgb_value: true,
        };
        color.profile.hue_sat_map = [Some(deep.clone()), Some(deep.clone())];
        let d = lcc_raw::dcp::parse(&write_dcp("SONY ILCE-7M4", "Deep", "A Contributor", &color).unwrap()).unwrap();
        let got = d.color.profile.hue_sat_map[0].as_ref().unwrap();
        assert_eq!((got.val_divisions, got.srgb_value, got.data.len()), (4, true, 48));
        assert!((got.data[47][0] - 4.7).abs() < 1e-4);
    }

    #[test]
    fn attribution_row_is_added_once_and_replaced() {
        let table = "| Path | x |\n|---|---|\n| `assets/camera-profiles/A.json` | a |\n| `other` | b |\n";
        let cal = sample();
        let once = with_attribution_row(table, "crates/camdb/data/calibrations/SONY_ILCE-7M4.json", &cal, "2026-10-25");
        let twice = with_attribution_row(&once, "crates/camdb/data/calibrations/SONY_ILCE-7M4.json", &cal, "2026-10-26");
        assert_eq!(twice.matches("SONY_ILCE-7M4.json").count(), 1);
        assert!(twice.contains("2026-10-26"));
        let lines: Vec<&str> = twice.lines().collect();
        assert!(lines[3].contains("SONY_ILCE-7M4.json"), "row goes after the camera-profile rows");
    }

    fn sample() -> Calibration {
        let fm = Mat3([[0.70, 0.18, 0.08], [0.28, 0.80, -0.08], [0.02, -0.15, 0.95]]);
        let cm = Mat3([[0.9, -0.3, -0.05], [-0.5, 1.4, 0.1], [-0.1, 0.3, 0.6]]);
        Calibration {
            version: calibration::VERSION,
            make: "SONY".into(),
            model: "ILCE-7M4".into(),
            aliases: vec![],
            lens: None,
            editor: Default::default(),
            method: calibration::METHOD.into(),
            tool_version: "0".into(),
            contributor: Some("A Contributor".into()),
            licence: calibration::LICENCE.into(),
            camera: calibration::CameraModel { illuminants: [17, 21], color_matrix: [cm, cm], forward_matrix: [fm, fm], hue_sat_map: None },
            looks: vec![],
            evidence: calibration::Evidence { photos: 24, ..Default::default() },
        }
    }
}
