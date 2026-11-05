//! `<cli> flat-field REFERENCE.raw [--grid N] [--smooth S] [--check B.raw [--force]] [--save MAP.json] [--json]`:
//! measure a flat-field reference (a raw shot through a diffuser, Capture One's LCC "Uniform Light"), print its
//! light fall-off, its optical centre and the lens database's vignetting profile for the same shot, and with
//! `--check` how much fall-off the map leaves in a second reference of the same setup ([`lcc_raw::flat_field`]).
//!
//! Every fall-off is in EV, positive = darker than the brightest part of the frame. Radii are measured from the
//! centre of the developed frame (the default crop, unrotated) in units of its half diagonal: the frame the lens
//! profile is expressed in and `Warp::gain` evaluates it in.

use std::path::Path;

use lcc_geom::{Orientation, Point};
use lcc_raw::flat_field::{self, CentreFit, CheckReport, FlatFieldMap, FlatFieldOptions, FlatFieldSource, SensorFrame};
use serde_json::{Value, json};

/// Radii of the radial table, in % of the half diagonal.
const RADII: [u32; 5] = [0, 25, 50, 75, 100];
/// The lens profile's ratio is shown only where the profile's own fall-off is at least this (EV).
const MIN_PROFILE_EV: f64 = 0.1;
/// Focal lengths and apertures this close count as the same.
const SAME: f64 = 0.05;

pub fn run(args: &[String]) -> Result<(), String> {
    let mut opts = FlatFieldOptions::default();
    let (mut files, mut check, mut save) = (Vec::new(), None, None);
    let (mut force, mut json_out) = (false, false);
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| -> Result<String, String> { it.next().cloned().ok_or(format!("{name} needs a value")) };
        match arg.as_str() {
            "--grid" => {
                let v = value("--grid")?;
                opts.long_cells = v.parse().map_err(|_| format!("--grid: `{v}` is not a number of cells"))?;
            }
            "--smooth" => {
                let v = value("--smooth")?;
                opts.smoothing = v.parse().map_err(|_| format!("--smooth: `{v}` is not a number of cells"))?;
            }
            "--check" => check = Some(value("--check")?),
            "--save" => save = Some(value("--save")?),
            "--force" => force = true,
            "--json" => json_out = true,
            a if a.starts_with("--") => return Err(format!("unknown option `{a}`")),
            f => files.push(f.to_string()),
        }
    }
    let [reference] = files.as_slice() else {
        return Err("flat-field needs one reference raw: REFERENCE.raw".into());
    };
    let mut a = Shot::load(reference)?;
    let map = flat_field::build(&a.normalized, a.source.clone(), &opts).map_err(|e| format!("{reference}: {e}"))?;
    // (the samples aren't needed any more; a second reference is decoded next)
    a.normalized.data = Vec::new();
    if let Some(out) = &save {
        let text = serde_json::to_string(&map).map_err(|e| e.to_string())?;
        std::fs::write(out, text).map_err(|e| format!("{out}: {e}"))?;
    }
    let measured = Profile::of(&a.frame, &|p| map.ev_at(p.x, p.y));
    let centre = flat_field::fit_centre(&map.ev_grid());
    let lens = lens_profile(&a);
    let checked = match &check {
        Some(path) => Some(check_against(&map, &a, path, &opts, force)?),
        None => None,
    };
    if json_out {
        let v = json!({
            "file": reference,
            "source": map.source,
            "sensor": map.sensor,
            "grid": map.grid,
            "smoothing": opts.smoothing,
            "falloff": measured.json(),
            "centre": centre,
            "lensProfile": match &lens {
                Ok((name, p)) => json!({"name": name, "falloff": p.json(), "ratio": ratios(&measured, p)}),
                Err(why) => json!({"none": why}),
            },
            "check": checked.as_ref().map(Checked::json),
            "saved": save,
        });
        println!("{}", serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?);
        return Ok(());
    }
    println!("Reference: {reference}");
    println!("  {}", describe(&map.source));
    println!(
        "  active area {} × {} px, grid {} × {} cells, smoothing σ {} cell(s)",
        map.sensor[0], map.sensor[1], map.grid[0], map.grid[1], opts.smoothing
    );
    println!("Fall-off (EV, positive = darker than the brightest part):");
    print_corners(&measured);
    print_centre(centre.as_ref());
    match &lens {
        Ok((name, p)) => {
            println!("Radial, from the frame centre (r = % of the half diagonal; lens profile: {name}):");
            println!("     r   measured   min–max          profile   measured/profile");
            for ((m, p), ratio) in measured.rings.iter().zip(&p.rings).zip(ratios(&measured, p)) {
                let ratio = ratio.map_or("–".to_string(), |r| format!("{r:.2}"));
                println!("  {:>3} %  {:>8.3}   {:>6.3}–{:<6.3}   {:>8.3}   {ratio}", m.pct, m.mean, m.min, m.max, p.mean);
            }
        }
        Err(why) => {
            println!("Radial, from the frame centre (r = % of the half diagonal; lens profile: {why}):");
            println!("     r   measured   min–max");
            for m in &measured.rings {
                println!("  {:>3} %  {:>8.3}   {:>6.3}–{:<6.3}", m.pct, m.mean, m.min, m.max);
            }
        }
    }
    println!("  (a large min–max spread at one radius means the fall-off is off-centre)");
    if let Some(c) = &checked {
        c.print();
    }
    if let Some(out) = &save {
        println!("Map saved: {out}");
    }
    Ok(())
}

/// A decoded reference.
struct Shot {
    path: String,
    normalized: lcc_raw::Normalized,
    frame: SensorFrame,
    meta: lcc_meta::Metadata,
    source: FlatFieldSource,
}

impl Shot {
    fn load(path: &str) -> Result<Shot, String> {
        let bytes = std::fs::read(Path::new(path)).map_err(|e| format!("{path}: {e}"))?;
        let mut sha = lcc_hash::Sha1::new();
        sha.update(&bytes);
        let sha1 = sha.finish().to_hex();
        let raw = lcc_raw::decode(&bytes).map_err(|e| format!("{path}: {e}"))?;
        // the file's metadata (as the lens lookup reads it), with the decoder's for whatever it lacks
        let mut meta = lcc_meta::extract(&bytes);
        let r = &raw.metadata;
        meta.make = meta.make.or_else(|| r.make.clone());
        meta.model = meta.model.or_else(|| r.model.clone());
        meta.serial_number = meta.serial_number.or_else(|| r.serial_number.clone());
        meta.lens_make = meta.lens_make.or_else(|| r.lens_make.clone());
        meta.lens_model = meta.lens_model.or_else(|| r.lens_model.clone());
        meta.lens_spec = meta.lens_spec.or(r.lens_spec);
        meta.focal_length = meta.focal_length.or(r.focal_length);
        meta.focal_length_35mm = meta.focal_length_35mm.or(r.focal_length_35mm);
        meta.f_number = meta.f_number.or(r.f_number);
        drop(bytes);
        let normalized = raw.normalized().map_err(|e| format!("{path}: {e}"))?;
        let frame = SensorFrame::of(&raw);
        let source = FlatFieldSource::from_metadata(&meta, Some(sha1));
        Ok(Shot { path: path.to_string(), normalized, frame, meta, source })
    }
}

/// One radius of the radial table.
struct Ring {
    pct: u32,
    mean: f64,
    min: f64,
    max: f64,
}

/// A fall-off sampled at the developed frame's corners and radii.
struct Profile {
    /// Top-left, top-right, bottom-left, bottom-right of the unrotated frame.
    corners: [f64; 4],
    rings: Vec<Ring>,
}

impl Profile {
    /// Sample `ev` (a function of sensor coordinates) over `frame`'s developed picture.
    fn of(frame: &SensorFrame, ev: &dyn Fn(Point) -> f64) -> Profile {
        Profile::in_developed(frame, &|p| ev(frame.cropped_to_sensor(p)))
    }

    /// Sample `ev` (a function of developed-frame coordinates).
    fn in_developed(frame: &SensorFrame, ev: &dyn Fn(Point) -> f64) -> Profile {
        let (w, h) = (frame.developed[0] as f64, frame.developed[1] as f64);
        let corners = [ev(Point::new(0.0, 0.0)), ev(Point::new(w, 0.0)), ev(Point::new(0.0, h)), ev(Point::new(w, h))];
        let (cx, cy, hd) = (w / 2.0, h / 2.0, w.hypot(h) / 2.0);
        let rings = RADII
            .iter()
            .map(|&pct| {
                let values: Vec<f64> = match pct {
                    0 => vec![ev(Point::new(cx, cy))],
                    // only the corners are this far out
                    100 => corners.to_vec(),
                    _ => (0..360)
                        .filter_map(|k| {
                            let t = (k as f64).to_radians();
                            let r = hd * pct as f64 / 100.0;
                            let p = Point::new(cx + r * t.cos(), cy + r * t.sin());
                            (p.x >= 0.0 && p.x <= w && p.y >= 0.0 && p.y <= h).then(|| ev(p))
                        })
                        .collect(),
                };
                let n = values.len().max(1) as f64;
                Ring {
                    pct,
                    mean: values.iter().sum::<f64>() / n,
                    min: values.iter().copied().fold(f64::INFINITY, f64::min),
                    max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                }
            })
            .collect();
        Profile { corners, rings }
    }

    fn json(&self) -> Value {
        let [tl, tr, bl, br] = self.corners;
        json!({
            "corners": {"topLeft": tl, "topRight": tr, "bottomLeft": bl, "bottomRight": br},
            "radial": self.rings.iter().map(|r| json!({"rPct": r.pct, "mean": r.mean, "min": r.min, "max": r.max})).collect::<Vec<_>>(),
        })
    }
}

/// measured / profile at each radius, where the profile's fall-off is at least [`MIN_PROFILE_EV`].
fn ratios(measured: &Profile, profile: &Profile) -> Vec<Option<f64>> {
    measured.rings.iter().zip(&profile.rings).map(|(m, p)| (p.mean >= MIN_PROFILE_EV).then(|| m.mean / p.mean)).collect()
}

/// The lens database's vignetting for the shot, sampled where the measured fall-off is, or why there is none.
///
/// Evaluated as `Warp::gain` evaluates it (`crates/pipeline/src/optics.rs`): the app looks the profile up for the
/// developed size and the file's orientation (`lens_profiles::resolve_raw` → `lookup_with`); here the same lookup
/// with `Orientation::Normal`, the same profile before `reorient_lens` re-expresses it for the rotated frame. The
/// profile's own radius (lensfun's `pa` model: 1 at half the calibration frame's diagonal) is converted by
/// `Profile::correction` (`crates/lensdb/src/lib.rs`) to a DNG `FixVignetteRadial` with 1 at this frame's half
/// diagonal, centred on it; its gain is `embedded_vignette_gain` at `Warp::gain`'s full strength, and the fall-off
/// in EV is its log2.
fn lens_profile(shot: &Shot) -> Result<(String, Profile), String> {
    // the database folder chosen in the app's Settings, as the app loads it at start (the environment still wins)
    let _settings = lcc_engine::lens_download::LensSettings::load(lcc_engine::config::config_dir());
    let db = lcc_engine::lens_profiles::database().ok_or("no lens database configured (Settings ▸ Folders ▸ Lens profiles)")?;
    let (w, h) = (shot.frame.developed[0] as f64, shot.frame.developed[1] as f64);
    let lens = shot.source.lens.clone().unwrap_or_else(|| "the lens".into());
    let at = format!(
        "{} at {} f/{}",
        lens,
        shot.source.focal_length.map_or("? mm".into(), |f| format!("{f} mm")),
        shot.source.aperture.map_or("?".into(), |a| a.to_string())
    );
    let (correction, applied) = lcc_engine::lens_profiles::lookup_with(&db, &shot.meta, None, w, h, Orientation::Normal)
        .ok_or(format!("no profile in the lens database for {at}"))?;
    let v = correction.vignette.ok_or(format!("the lens database's profile has no vignetting for {at}"))?;
    let name = match applied {
        lcc_engine::lens_profiles::Applied::Found(n) | lcc_engine::lens_profiles::Applied::Chosen(n) => n,
        lcc_engine::lens_profiles::Applied::Embedded => lens,
    };
    let ev = |p: Point| lcc_pipeline::optics::embedded_vignette_gain(&v, p, w, h).max(1e-9).log2();
    Ok((name, Profile::in_developed(&shot.frame, &ev)))
}

/// The second reference of `--check`, and what the map leaves of its fall-off.
struct Checked {
    path: String,
    source: FlatFieldSource,
    forced: Vec<String>,
    before: Profile,
    after: Profile,
    centre: Option<CentreFit>,
    report: CheckReport,
}

fn check_against(map: &FlatFieldMap, a: &Shot, path: &str, opts: &FlatFieldOptions, force: bool) -> Result<Checked, String> {
    let b = Shot::load(path)?;
    let size = [b.normalized.width, b.normalized.height];
    if size != map.sensor {
        return Err(format!("{path}: the active area is {} × {} px, the reference's {} × {} px", size[0], size[1], map.sensor[0], map.sensor[1]));
    }
    let differs = differences(&a.source, &b.source);
    if !differs.is_empty() && !force {
        return Err(format!("{path} is not the reference's setup ({}); --force compares anyway", differs.join(", ")));
    }
    let cells = flat_field::cell_means(&b.normalized, map.grid, opts).map_err(|e| format!("{path}: {e}"))?;
    let report = flat_field::check(map, &cells, opts).map_err(|e| format!("{path}: {e}"))?;
    let before = Profile::of(&b.frame, &|p| report.before.at(p.x, p.y));
    let after = Profile::of(&b.frame, &|p| report.after.at(p.x, p.y));
    let centre = flat_field::fit_centre(&report.before);
    Ok(Checked { path: b.path, source: b.source, forced: differs, before, after, centre, report })
}

/// What differs between two references' setups: camera, lens, focal length, aperture.
fn differences(a: &FlatFieldSource, b: &FlatFieldSource) -> Vec<String> {
    let mut out = Vec::new();
    let text = |s: &Option<String>| s.as_deref().unwrap_or("?").to_string();
    if (&a.make, &a.model) != (&b.make, &b.model) {
        out.push(format!("camera {} {} vs {} {}", text(&a.make), text(&a.model), text(&b.make), text(&b.model)));
    }
    if a.lens != b.lens {
        out.push(format!("lens {} vs {}", text(&a.lens), text(&b.lens)));
    }
    let near = |x: Option<f64>, y: Option<f64>| match (x, y) {
        (Some(x), Some(y)) => (x - y).abs() <= SAME,
        (None, None) => true,
        _ => false,
    };
    let num = |v: Option<f64>| v.map_or("?".into(), |v| v.to_string());
    if !near(a.focal_length, b.focal_length) {
        out.push(format!("focal length {} vs {} mm", num(a.focal_length), num(b.focal_length)));
    }
    if !near(a.aperture, b.aperture) {
        out.push(format!("aperture f/{} vs f/{}", num(a.aperture), num(b.aperture)));
    }
    out
}

impl Checked {
    fn json(&self) -> Value {
        json!({
            "file": self.path,
            "source": self.source,
            "forced": self.forced,
            "before": self.before.json(),
            "after": self.after.json(),
            "centre": self.centre,
            "maxAbsResidual": self.report.max_abs,
            "peakToPeak": self.report.peak_to_peak,
            "pass": self.report.passes(),
        })
    }

    fn print(&self) {
        println!("Check: {}", self.path);
        println!("  {}", describe(&self.source));
        if !self.forced.is_empty() {
            println!("  compared despite: {}", self.forced.join(", "));
        }
        println!("  Its own fall-off before correction (EV, from its brightest smoothed region):");
        print_corners(&self.before);
        print_centre(self.centre.as_ref());
        println!("  After the map at amount 1.0 (EV from the median corrected cell, positive = darker):");
        print_corners(&self.after);
        println!("     r   before   after (mean, min–max)");
        for (b, a) in self.before.rings.iter().zip(&self.after.rings) {
            println!("  {:>3} %  {:>6.3}   {:>+7.3}  {:>+7.3}–{:<+7.3}", b.pct, b.mean, a.mean, a.min, a.max);
        }
        let verdict = if self.report.passes() { "PASS" } else { "FAIL" };
        println!(
            "  max |residual| {:.3} EV, peak-to-peak {:.3} EV: {verdict} (needs < 0.05 and < 0.1)",
            self.report.max_abs, self.report.peak_to_peak
        );
    }
}

fn print_corners(p: &Profile) {
    let [tl, tr, bl, br] = p.corners;
    println!("  corners: top-left {tl:.3}  top-right {tr:.3}  bottom-left {bl:.3}  bottom-right {br:.3}  (unrotated frame)");
}

fn print_centre(c: Option<&CentreFit>) {
    match c {
        Some(c) => println!("  optical centre: {:+.1} % x, {:+.1} % y from the frame centre (fit RMS {:.4} EV)", c.dx_pct, c.dy_pct, c.rms_ev),
        None => println!("  optical centre: no fit"),
    }
}

fn describe(s: &FlatFieldSource) -> String {
    let text = |v: &Option<String>| v.clone().unwrap_or_else(|| "?".into());
    format!(
        "{} {} (serial {}) · {} · {} · f/{}",
        text(&s.make),
        text(&s.model),
        text(&s.serial),
        text(&s.lens),
        s.focal_length.map_or("? mm".into(), |f| format!("{f} mm")),
        s.aperture.map_or("?".into(), |a| a.to_string())
    )
}
