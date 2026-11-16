//! Hue/saturation tables (DNG `HueSatMap` / `LookTable`, ProPhoto HSV) fitted by kernel regression with a smoothness
//! term: each sample's needed correction is splatted at its table input, then the grid is solved by Gauss–Seidel.
use lcc_raw::profile::HsvTable;

pub const HUES: usize = 90;
pub const SATS: usize = 30;
/// A cell's correction counts fully once the cells within [`SUPPORT_REACH`] steps (hue and saturation) hold clearly
/// more than [`SUPPORT_SHARE`] of all the sample weight, and fades towards none as that goes to 0.
const SUPPORT_REACH: usize = 2;
const SUPPORT_SHARE: f64 = 0.002;

/// HSV of a linear RGB colour as the DNG tables read it (hue in degrees).
pub fn hsv(c: [f32; 3]) -> Option<(f64, f64, f64)> {
    let max = c[0].max(c[1]).max(c[2]);
    if !(max.is_finite() && max > 1e-6) {
        return None;
    }
    let min = c[0].min(c[1]).min(c[2]);
    let d = max - min;
    let h = if d <= 0.0 {
        0.0
    } else if max == c[0] {
        ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if max == c[1] {
        (c[2] - c[0]) / d + 2.0
    } else {
        (c[0] - c[1]) / d + 4.0
    };
    Some((f64::from(h) * 60.0, f64::from(d / max), f64::from(max)))
}

pub fn identity() -> HsvTable {
    HsvTable { hue_divisions: HUES, sat_divisions: SATS, val_divisions: 1, data: vec![[0.0, 1.0, 1.0]; HUES * SATS], srgb_value: false }
}

fn wrap180(d: f64) -> f64 {
    (d + 180.0).rem_euclid(360.0) - 180.0
}

/// One photo's accumulated weight and weighted correction in one cell (kept only with a per-photo cap).
type PhotoShare<const N: usize> = (usize, f64, [f64; N]);

/// Add `w` and `wv` to `photo`'s share of a cell.
fn add_share<const N: usize>(cell: &mut Vec<PhotoShare<N>>, photo: usize, w: f64, wv: [f64; N]) {
    if let Some(e) = cell.iter_mut().find(|e| e.0 == photo) {
        e.1 += w;
        for i in 0..N {
            e.2[i] += wv[i];
        }
    } else {
        cell.push((photo, w, wv));
    }
}

/// The per-photo cap ([`ValueShape::photo_cap`]): every cell's weight and weighted correction with no photo holding
/// more than `cap` of the cell's (capped) weight, so one photo with many samples in a cell can't decide that cell on
/// its own. A photo's weight in a cell is cut to the level L with L = cap × Σ min(wᵢ, L), but never below cap × the
/// mean cell weight, so a cell only one or two photos reach keeps a sensible share of its data.
fn capped<const N: usize>(per: &[Vec<PhotoShare<N>>], cap: f64) -> (Vec<f64>, Vec<[f64; N]>) {
    let total: f64 = per.iter().flatten().map(|e| e.1).sum();
    let floor = cap * total / per.len().max(1) as f64;
    let mut w = vec![0.0; per.len()];
    let mut v = vec![[0.0; N]; per.len()];
    for (k, cell) in per.iter().enumerate() {
        let mut limit = cell.iter().map(|e| e.1).fold(0.0, f64::max);
        for _ in 0..60 {
            let next = (cap * cell.iter().map(|e| e.1.min(limit)).sum::<f64>()).max(floor);
            if (next - limit).abs() <= 1e-12 * limit.max(1e-300) {
                break;
            }
            limit = next.min(limit);
        }
        for e in cell {
            if e.1 <= 0.0 {
                continue;
            }
            let keep = e.1.min(limit) / e.1;
            w[k] += e.1 * keep;
            for i in 0..N {
                v[k][i] += e.2[i] * keep;
            }
        }
    }
    (w, v)
}

/// A cell counts as holding real data for [`curvature_weights`] at this share of the mean weight of cells with data.
const CURVATURE_SUPPORT: f64 = 0.05;

/// The weight of the curvature term ([`ValueShape::curvature`]) of the second difference centred on each cell, along
/// hue (`along_hue`) or saturation, on one `HUES × SATS` plane of weights `w` (starting at `base`). Where all three
/// cells it spans hold real data (each at least [`CURVATURE_SUPPORT`] × the plane's mean weight of cells with data):
/// `lam` × the LARGEST of the three weights, so a densely sampled cell whose samples disagree with its neighbours'
/// can't outweigh its own smoothness (a post-smoothing of the table treats every cell alike, and that is what removed
/// the potholes; weighted by the mean weight, as the Laplacian term is, or by the cell's own, a dense pothole only
/// spreads into a broad dip). Where the second difference reaches into cells with little or no data: `lam` × the
/// smallest of the three, so the edge of the data isn't tied to empty cells whose values only the weak Laplacian sets.
fn curvature_weights(w: &[f64], base: usize, lam: f64, along_hue: bool) -> Vec<f64> {
    let at = |h: usize, s: usize| w.get(base + h * SATS + s).copied().unwrap_or(0.0);
    let filled: Vec<f64> = (0..HUES * SATS).map(|k| at(k / SATS, k % SATS)).filter(|x| *x > 0.0).collect();
    let floor = CURVATURE_SUPPORT * filled.iter().sum::<f64>() / filled.len().max(1) as f64;
    let mut out = vec![0.0; HUES * SATS];
    for h in 0..HUES {
        for s in 0..SATS {
            let three = if along_hue {
                [at((h + HUES - 1) % HUES, s), at(h, s), at((h + 1) % HUES, s)]
            } else if s == 0 || s + 1 >= SATS {
                continue;
            } else {
                [at(h, s - 1), at(h, s), at(h, s + 1)]
            };
            let lo = three.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = three.iter().copied().fold(0.0, f64::max);
            out[h * SATS + s] = lam * if floor > 0.0 && lo >= floor { hi } else { lo };
        }
    }
    out
}

/// The curvature terms holding cell (h, s) of one plane, for component `i` of `x` (plane starting at `base`), in the
/// Gauss–Seidel form (num −= wt·α·β, den += wt·α²): the second differences along hue centred on h−1, h, h+1, and along
/// saturation centred on s−1, s, s+1 where both neighbours exist.
fn curvature_terms<const N: usize>(x: &[[f64; N]], base: usize, h: usize, s: usize, i: usize, wh: &[f64], ws: &[f64]) -> (f64, f64) {
    let at = |hh: usize, ss: usize| x.get(base + (hh % HUES) * SATS + ss).map_or(0.0, |p| p[i]);
    let (mut num, mut den) = (0.0, 0.0);
    // along hue (circular)
    for (c, alpha, beta) in [
        ((h + HUES - 1) % HUES, 1.0, at(h + HUES - 2, s) - 2.0 * at(h + HUES - 1, s)),
        (h, -2.0, at(h + HUES - 1, s) + at(h + 1, s)),
        ((h + 1) % HUES, 1.0, -2.0 * at(h + 1, s) + at(h + 2, s)),
    ] {
        let wt = wh.get(c * SATS + s).copied().unwrap_or(0.0);
        num -= wt * alpha * beta;
        den += wt * alpha * alpha;
    }
    // along saturation (only second differences with both neighbours on the grid)
    for dc in [-1i64, 0, 1] {
        let c = s as i64 + dc;
        if c < 1 || c + 1 >= SATS as i64 {
            continue;
        }
        let c = c as usize;
        let (alpha, beta) = match dc {
            0 => (-2.0, at(h, c - 1) + at(h, c + 1)),
            -1 => (1.0, at(h, c - 1) - 2.0 * at(h, c)),
            _ => (1.0, -2.0 * at(h, c) + at(h, c + 1)),
        };
        let wt = ws.get(h * SATS + c).copied().unwrap_or(0.0);
        num -= wt * alpha * beta;
        den += wt * alpha * alpha;
    }
    (num, den)
}

/// Accumulated corrections (hue shift °, ln saturation ratio, ln value ratio) on the table grid.
#[derive(Clone, Debug)]
pub struct Splat {
    w: Vec<f64>,
    v: Vec<[f64; 3]>,
    /// [`ValueShape::photo_cap`] (0: off) and, with it on, every photo's share of every cell.
    cap: f64,
    per: Vec<Vec<PhotoShare<3>>>,
}

impl Default for Splat {
    fn default() -> Self {
        Splat { w: vec![0.0; HUES * SATS], v: vec![[0.0; 3]; HUES * SATS], cap: 0.0, per: Vec::new() }
    }
}

impl Splat {
    /// An empty splat with the per-photo cap `cap` ([`ValueShape::photo_cap`]; 0 or not below 1: off). Samples must then
    /// come through [`Splat::add_from`] to be told apart by photo.
    pub fn with_cap(cap: f64) -> Self {
        let mut s = Splat::default();
        if cap.is_finite() && cap > 0.0 && cap < 1.0 {
            s.cap = cap;
            s.per = vec![Vec::new(); HUES * SATS];
        }
        s
    }

    /// Add one sample: the table input `input`, the colour the table currently gives `out` and the wanted one
    /// `target` (all linear ProPhoto), with weight `weight`.
    #[cfg(test)]
    pub fn add(&mut self, input: [f32; 3], out: [f32; 3], target: [f32; 3], weight: f64) {
        self.add_from(0, input, out, target, weight);
    }

    /// [`Splat::add`] for a sample of photo `photo` (only the per-photo cap tells photos apart).
    pub fn add_from(&mut self, photo: usize, input: [f32; 3], out: [f32; 3], target: [f32; 3], weight: f64) {
        let (Some((hi, si, _)), Some((ho, so, vo)), Some((ht, st, vt))) = (hsv(input), hsv(out), hsv(target)) else {
            return;
        };
        if weight.is_nan() || weight <= 0.0 || so < 0.02 || st < 0.0 {
            return;
        }
        // hue is meaningless for near-neutral colours: they correct value only, a little saturation
        let hue_weight = ((so.min(st) - 0.02) / 0.08).clamp(0.0, 1.0);
        let corr = [wrap180(ht - ho) * hue_weight, (st.max(1e-4) / so).ln().clamp(-1.0, 1.0), (vt / vo).ln().clamp(-1.0, 1.0)];
        let hf = hi / 360.0 * HUES as f64;
        let h0 = (hf.floor() as isize).rem_euclid(HUES as isize) as usize;
        let th = hf - hf.floor();
        let sf = si.clamp(0.0, 1.0) * (SATS - 1) as f64;
        let s0 = (sf as usize).min(SATS - 2);
        let ts = sf - s0 as f64;
        for (dh, wh) in [(0, 1.0 - th), (1, th)] {
            for (ds, ws) in [(0, 1.0 - ts), (1, ts)] {
                let k = ((h0 + dh) % HUES) * SATS + s0 + ds;
                let w = weight * wh * ws;
                if let (Some(cw), Some(cv)) = (self.w.get_mut(k), self.v.get_mut(k)) {
                    *cw += w;
                    for i in 0..3 {
                        cv[i] += w * corr[i];
                    }
                }
                if let Some(cell) = self.per.get_mut(k) {
                    add_share(cell, photo, w, corr.map(|c| w * c));
                }
            }
        }
    }

    pub fn total(&self) -> f64 {
        self.w.iter().sum()
    }

    /// The smooth correction grid: data term, Laplacian smoothness (`smooth` × mean cell weight) and a weak pull
    /// to no correction where there is no data.
    #[cfg(test)]
    pub fn solve(&self, smooth: f64) -> Vec<[f64; 3]> {
        self.solve_with(smooth, 0.0)
    }

    /// [`Splat::solve`] with the curvature term `curvature` ([`ValueShape::curvature`]; 0: off) and, when this splat
    /// was made [`Splat::with_cap`], the per-photo cap applied to the cells first. With both off it is
    /// [`Splat::solve`] exactly.
    pub fn solve_with(&self, smooth: f64, curvature: f64) -> Vec<[f64; 3]> {
        let n = HUES * SATS;
        let total = self.total();
        let mut x = vec![[0.0; 3]; n];
        if total.is_nan() || total <= 0.0 {
            return x;
        }
        let capped_cells = (!self.per.is_empty()).then(|| capped(&self.per, self.cap));
        let (cw, cv) = match &capped_cells {
            Some((w, v)) => (w.as_slice(), v.as_slice()),
            None => (self.w.as_slice(), self.v.as_slice()),
        };
        let kappa = smooth * total / n as f64;
        let eps = 1e-3 * total / n as f64;
        let curved = curvature.is_finite() && curvature > 0.0;
        let (wh, ws) = if curved {
            (curvature_weights(cw, 0, curvature, true), curvature_weights(cw, 0, curvature, false))
        } else {
            (Vec::new(), Vec::new())
        };
        // with the curvature term: the plain solve first (a stiff curvature term from a zero start settles its broad,
        // low-frequency part far too slowly), then sweeps with the term on top of that
        let sweeps = if curved { 400 + 1500 } else { 400 };
        for sweep in 0..sweeps {
            let curved_now = curved && sweep >= 400;
            for h in 0..HUES {
                for s in 0..SATS {
                    let k = h * SATS + s;
                    let mut nb = vec![((h + 1) % HUES) * SATS + s, ((h + HUES - 1) % HUES) * SATS + s];
                    if s > 0 {
                        nb.push(k - 1);
                    }
                    if s + 1 < SATS {
                        nb.push(k + 1);
                    }
                    let (w, v) = (cw.get(k).copied().unwrap_or(0.0), cv.get(k).copied().unwrap_or([0.0; 3]));
                    let den = w + kappa * nb.len() as f64 + eps;
                    let mut new = [0.0; 3];
                    for i in 0..3 {
                        let sum: f64 = nb.iter().map(|j| x.get(*j).map_or(0.0, |p: &[f64; 3]| p[i])).sum();
                        if curved_now {
                            let (cn, cd) = curvature_terms(&x, 0, h, s, i, &wh, &ws);
                            new[i] = (v[i] + kappa * sum + cn) / (den + cd);
                        } else {
                            new[i] = (v[i] + kappa * sum) / den;
                        }
                    }
                    if let Some(p) = x.get_mut(k) {
                        *p = new;
                    }
                }
            }
        }
        // Where hardly any sample lies nearby, smoothing alone carries a few odd samples' correction across a whole
        // empty region, and each closed-loop round adds to it (a 5-photo Lumix G9 fit turned saturated blues and
        // purples by up to 117° from 0.05 % of the sample weight). Fade each cell's correction by the data around it.
        let w0 = SUPPORT_SHARE * total;
        for (k, p) in x.iter_mut().enumerate() {
            let (h, s) = (k / SATS, k % SATS);
            let mut near = 0.0;
            for dh in 0..=2 * SUPPORT_REACH {
                for ds in 0..=2 * SUPPORT_REACH {
                    let (Some(s2), hh) = ((s + ds).checked_sub(SUPPORT_REACH), (h + HUES + dh - SUPPORT_REACH) % HUES) else { continue };
                    if s2 < SATS {
                        near += self.w.get(hh * SATS + s2).copied().unwrap_or(0.0);
                    }
                }
            }
            let keep = near * near / (near * near + w0 * w0);
            for v in p.iter_mut() {
                *v *= keep;
            }
        }
        x
    }
}

/// `table` with `corr` composed on top (hue shifts add, scales multiply), clamped to the format's bounds.
pub fn compose(table: &HsvTable, corr: &[[f64; 3]], shrink: f64) -> HsvTable {
    let mut t = table.clone();
    for (e, c) in t.data.iter_mut().zip(corr) {
        let k = shrink.clamp(0.0, 1.0);
        e[0] = (f64::from(e[0]) + k * c[0]).clamp(-170.0, 170.0) as f32;
        e[1] = (f64::from(e[1]) * (k * c[1]).exp()).clamp(0.0, 8.0) as f32;
        e[2] = (f64::from(e[2]) * (k * c[2]).exp()).clamp(0.0, 8.0) as f32;
    }
    t
}

/// Whether a table changes anything noticeable.
pub fn is_identity(t: &HsvTable) -> bool {
    t.data.iter().all(|e| e[0].abs() < 1e-3 && (e[1] - 1.0).abs() < 1e-4 && (e[2] - 1.0).abs() < 1e-4)
}

// ---- Value divisions (`--value-divisions N`, N > 1) ----------------------------------------------------------------
//
// A map with value divisions is the single-step map plus a brightness-dependent part: every value plane starts as the
// single-step correction, and a smooth deviation in hue shift and saturation scale is fitted on top. The deviation is
// pinned to zero on the first and last plane (black and white keep the single-step correction), follows a curve with
// a strong curvature penalty along value, is smoothed across hue and saturation like the single-step map, and is
// pulled to zero where few samples back it, so thin data falls back to the single-step correction. The value axis is
// sRGB-encoded (`ProfileHueSatMapEncoding` = 1), as the renderer indexes it (`HsvTable::apply`). Only maps with
// value divisions go through these functions; the single-step path above is unchanged.

/// Smoothness across hue and saturation of the deviation (× mean cell weight), as [`Splat::solve`]'s.
const VALUE_SMOOTH: f64 = 0.5;
/// Curvature penalty along value (× mean cell weight): the deviation bends slowly with brightness. Cells with more
/// data than the mean follow it; sparse cells follow the curve of their neighbours along value.
const VALUE_CURVE: f64 = 2.0;
/// Pull of the deviation toward zero (× mean cell weight): a cell needs about this much weight before its own data
/// moves it halfway away from the single-step correction.
const VALUE_PULL: f64 = 1.0;

/// How the brightness-dependent part ends toward white (black always keeps the single-step correction).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ValueEnding {
    /// Pinned to zero on the white plane: white keeps the single-step correction. The return to it in the last step
    /// is steep where the brights differ most from the single step.
    Pin,
    /// Pinned on the white plane, and pulled toward zero more strongly over the upper half of the value range
    /// ([`FADE_PULL`]), so the return to the single-step correction is spread over more brightness.
    Fade,
    /// Not pinned: the white plane holds the deviation of the planes below it (a first-difference tie of
    /// [`HOLD_TIE`]), so white keeps the last supported step's correction instead of returning to the single step
    /// (the default: no steep last step, and no edge at white in a gradient).
    #[default]
    Hold,
    /// [`ValueEnding::Hold`] for saturated colours and [`ValueEnding::Pin`] for near-neutral ones: on the white plane
    /// the hold tie weakens and a pin to the single-step correction takes over as saturation falls below
    /// [`HOLD_SAT_FULL`] (smoothly; fully pinned at grey). Bright near-neutrals (clouds, highlights) have few samples,
    /// and holding them carries a thinly supported deviation up to white.
    HoldSaturated,
}

impl ValueEnding {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pin" => Some(ValueEnding::Pin),
            "fade" => Some(ValueEnding::Fade),
            "hold" => Some(ValueEnding::Hold),
            "hold-sat" => Some(ValueEnding::HoldSaturated),
            _ => None,
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            ValueEnding::Pin => "pin",
            ValueEnding::Fade => "fade",
            ValueEnding::Hold => "hold",
            ValueEnding::HoldSaturated => "hold-sat",
        }
    }
}

/// [`ValueEnding::Fade`]: extra pull toward zero at white (× [`VALUE_PULL`]), growing linearly from the middle of the
/// value range.
const FADE_PULL: f64 = 8.0;
/// [`ValueEnding::Hold`]: weight of the first-difference tie between the white plane and the one below (× mean cell
/// weight).
const HOLD_TIE: f64 = 8.0;
/// [`ValueEnding::HoldSaturated`]: the saturation from which the white plane is held as by [`ValueEnding::Hold`].
pub const HOLD_SAT_FULL: f64 = 0.25;
/// [`ValueEnding::HoldSaturated`]: weight of the pin on the white plane at grey (× mean cell weight): large, so the
/// plane follows the single-step correction there.
const HOLD_SAT_PIN: f64 = 1000.0;

/// How the brightness-dependent part is shaped: its ending at white and two experimental options (both off at 0).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ValueShape {
    pub ending: ValueEnding,
    /// Below this input saturation the part fades toward the single-step correction ([`fade_low_saturation`]).
    pub low_sat_fade: f64,
    /// Extra stiffness along value over the upper half of the range: the curvature penalty grows to
    /// `1 + bright_smooth` times its usual weight at white.
    pub bright_smooth: f64,
    /// Fade the part toward the single-step correction where few samples lie nearby ([`ValueSplat::solve`]), as the
    /// single-step fit fades its own correction.
    pub support_fade: bool,
    /// Pull the part to the single-step correction for sky blues ([`pull_blue`]).
    pub blue_pull: bool,
    /// Squared second differences along hue and along saturation in every table solve (the single-step map, the look
    /// table and the brightness-dependent part), each weighted by this × the local sample weight ([`curvature_weights`]):
    /// 0 = off. On the Z 8 data 40 rendered about like a Gaussian post-smoothing of the table over 1.5
    /// cells and 320 like 3 cells.
    pub curvature: f64,
    /// Most of a cell's weight one photo may hold (0 = off), e.g. 0.25.
    pub photo_cap: f64,
}

impl From<ValueEnding> for ValueShape {
    fn from(ending: ValueEnding) -> Self {
        ValueShape { ending, ..Default::default() }
    }
}

/// Fade a brightness-dependent deviation grid (`HUES × SATS × vals`, as [`ValueSplat::solve`] gives) toward zero at low
/// saturation: × smoothstep(s / `below`), so grey and near-grey colours keep the single-step correction and the
/// deviation is whole from saturation `below` up. Nothing when `below` is 0 (or not a number).
pub fn fade_low_saturation(dev: &mut [[f64; 2]], below: f64) {
    if !(below.is_finite() && below > 0.0) {
        return;
    }
    for (k, d) in dev.iter_mut().enumerate() {
        let s = (k % SATS) as f64 / (SATS - 1) as f64;
        let t = (s / below).clamp(0.0, 1.0);
        let w = t * t * (3.0 - 2.0 * t);
        d[0] *= w;
        d[1] *= w;
    }
}

/// Input hues (linear ProPhoto HSV, degrees) where [`pull_blue`] removes the brightness-dependent part entirely: the
/// hues of cloud pixels in blue skies (measured on four sky photos rendered without hue/saturation map; their 2nd–98th
/// percentiles lie within about 200–240°, at saturation 0.17–0.46).
pub const BLUE_PULL_HUES: (f64, f64) = (200.0, 240.0);
/// Width (degrees) of the smooth taper on each side of [`BLUE_PULL_HUES`], from the full pull to none.
pub const BLUE_PULL_TAPER: f64 = 15.0;

/// How much of the brightness-dependent part [`pull_blue`] removes at input hue `h` (degrees): 1 inside
/// [`BLUE_PULL_HUES`], smoothstep down to 0 over [`BLUE_PULL_TAPER`] on each side.
pub fn blue_pull_weight(h: f64) -> f64 {
    let (lo, hi) = BLUE_PULL_HUES;
    let h = h.rem_euclid(360.0);
    let outside = if h < lo { lo - h } else if h > hi { h - hi } else { 0.0 };
    let t = (1.0 - outside / BLUE_PULL_TAPER).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Pull a brightness-dependent deviation grid (`vals × HUES × SATS`, as [`ValueSplat::solve`] gives) to the
/// single-step correction for sky blues, at every saturation and brightness: × (1 − [`blue_pull_weight`]). Clouds in a
/// blue sky are bright, fairly saturated blues with few training samples per brightness step, where a brightness
/// dependence learned from a few skies turns neighbouring cloud areas lavender or peach.
pub fn pull_blue(dev: &mut [[f64; 2]]) {
    for (k, d) in dev.iter_mut().enumerate() {
        let h = ((k / SATS) % HUES) as f64 * 360.0 / HUES as f64;
        let keep = 1.0 - blue_pull_weight(h);
        d[0] *= keep;
        d[1] *= keep;
    }
}

/// The sRGB-encoded value index the renderer reads a linear colour's table entry at.
pub fn encoded_value(c: [f32; 3]) -> Option<f64> {
    let (_, _, v) = hsv(c)?;
    Some(f64::from(lcc_color::transfer::linear_to_srgb((v as f32).min(1.0))))
}

/// The value plane nearest to encoded value `e` in a map with `vals` divisions.
pub fn value_step(e: f64, vals: usize) -> usize {
    let n = vals.max(2);
    ((e.clamp(0.0, 1.0) * (n - 1) as f64).round() as usize).min(n - 1)
}

/// `table` (single step) spread over `vals` value planes with an sRGB-encoded value axis, rendering the same colours:
/// hue shifts and saturation scales copied to every plane, the linear value scale converted to the encoded scale at
/// each plane's value.
pub fn with_value_divisions(table: &HsvTable, vals: usize) -> HsvTable {
    let vals = vals.max(2);
    let plane = table.hue_divisions * table.sat_divisions;
    let mut data = Vec::with_capacity(plane * vals);
    for v in 0..vals {
        let e = v as f32 / (vals - 1) as f32;
        let l = lcc_color::transfer::srgb_to_linear(e);
        for entry in table.data.iter().take(plane) {
            let k = entry[2];
            // the encoded scale that takes this plane's value where the linear scale takes it (12.92 slope at black);
            // not clipped at 1 here: the renderer clips the scaled value, and a clipped scale on the white plane would
            // lose the brightening of every value between it and the plane below
            let vs = if e <= 0.0 { k } else { lcc_color::transfer::linear_to_srgb(l * k) / e };
            data.push([entry[0], entry[1], if vs.is_finite() { vs.clamp(0.0, 8.0) } else { 1.0 }]);
        }
    }
    HsvTable { hue_divisions: table.hue_divisions, sat_divisions: table.sat_divisions, val_divisions: vals, data, srgb_value: true }
}

/// Accumulated brightness-dependent corrections (hue shift °, ln saturation ratio) on a `HUES × SATS × vals` grid.
#[derive(Clone, Debug)]
pub struct ValueSplat {
    vals: usize,
    w: Vec<f64>,
    v: Vec<[f64; 2]>,
    cap: f64,
    per: Vec<Vec<PhotoShare<2>>>,
}

impl ValueSplat {
    pub fn new(vals: usize) -> Self {
        let vals = vals.max(2);
        ValueSplat { vals, w: vec![0.0; HUES * SATS * vals], v: vec![[0.0; 2]; HUES * SATS * vals], cap: 0.0, per: Vec::new() }
    }

    /// [`ValueSplat::new`] with the per-photo cap `cap` ([`ValueShape::photo_cap`]; 0 or not below 1: off).
    pub fn with_cap(vals: usize, cap: f64) -> Self {
        let mut s = ValueSplat::new(vals);
        if cap.is_finite() && cap > 0.0 && cap < 1.0 {
            s.cap = cap;
            s.per = vec![Vec::new(); s.w.len()];
        }
        s
    }

    fn index(&self, v: usize, h: usize, s: usize) -> usize {
        (v * HUES + h) * SATS + s
    }

    /// Add one sample as [`Splat::add`], spread over the two value planes around its encoded value.
    #[cfg(test)]
    pub fn add(&mut self, input: [f32; 3], out: [f32; 3], target: [f32; 3], weight: f64) {
        self.add_from(0, input, out, target, weight);
    }

    /// [`ValueSplat::add`] for a sample of photo `photo` (only the per-photo cap tells photos apart).
    pub fn add_from(&mut self, photo: usize, input: [f32; 3], out: [f32; 3], target: [f32; 3], weight: f64) {
        let (Some((hi, si, _)), Some(e), Some((ho, so, _)), Some((ht, st, _))) = (hsv(input), encoded_value(input), hsv(out), hsv(target)) else {
            return;
        };
        if weight.is_nan() || weight <= 0.0 || so < 0.02 || st < 0.0 {
            return;
        }
        let hue_weight = ((so.min(st) - 0.02) / 0.08).clamp(0.0, 1.0);
        let corr = [wrap180(ht - ho) * hue_weight, (st.max(1e-4) / so).ln().clamp(-1.0, 1.0)];
        let hf = hi / 360.0 * HUES as f64;
        let h0 = (hf.floor() as isize).rem_euclid(HUES as isize) as usize;
        let th = hf - hf.floor();
        let sf = si.clamp(0.0, 1.0) * (SATS - 1) as f64;
        let s0 = (sf as usize).min(SATS - 2);
        let ts = sf - s0 as f64;
        let vf = e.clamp(0.0, 1.0) * (self.vals - 1) as f64;
        let v0 = (vf as usize).min(self.vals - 2);
        let tv = vf - v0 as f64;
        for (dv, wv) in [(0, 1.0 - tv), (1, tv)] {
            for (dh, wh) in [(0, 1.0 - th), (1, th)] {
                for (ds, ws) in [(0, 1.0 - ts), (1, ts)] {
                    let k = self.index(v0 + dv, (h0 + dh) % HUES, s0 + ds);
                    let w = weight * wv * wh * ws;
                    if let (Some(cw), Some(cv)) = (self.w.get_mut(k), self.v.get_mut(k)) {
                        *cw += w;
                        for i in 0..2 {
                            cv[i] += w * corr[i];
                        }
                    }
                    if let Some(cell) = self.per.get_mut(k) {
                        add_share(cell, photo, w, corr.map(|c| w * c));
                    }
                }
            }
        }
    }

    pub fn total(&self) -> f64 {
        self.w.iter().sum()
    }

    /// The smooth deviation grid (hue shift °, ln saturation ratio), zero on the first value plane and, unless
    /// `ending` is [`ValueEnding::Hold`], on the last. Gauss–Seidel on: data term + [`VALUE_SMOOTH`] Laplacian across
    /// hue/saturation + [`VALUE_CURVE`] squared second differences along value + [`VALUE_PULL`] toward zero (each ×
    /// mean cell weight), and the ending's own term.
    pub fn solve(&self, shape: impl Into<ValueShape>) -> Vec<[f64; 2]> {
        let shape: ValueShape = shape.into();
        let ending = shape.ending;
        let (vals, n) = (self.vals, self.w.len());
        let mut x = vec![[0.0; 2]; n];
        let total = self.total();
        if total.is_nan() || total <= 0.0 || vals < 3 {
            return x;
        }
        let mean = total / n as f64;
        let (kappa, curve, pull) = (VALUE_SMOOTH * mean, VALUE_CURVE * mean, VALUE_PULL * mean);
        // the per-photo cap and the curvature across hue and saturation (both opt-in; off, this is the plain solve)
        let capped_cells = (!self.per.is_empty()).then(|| capped(&self.per, self.cap));
        let (cw, cv) = match &capped_cells {
            Some((w, v)) => (w.as_slice(), v.as_slice()),
            None => (self.w.as_slice(), self.v.as_slice()),
        };
        let curved = shape.curvature.is_finite() && shape.curvature > 0.0;
        let plane_w: Vec<(Vec<f64>, Vec<f64>)> = if curved {
            (0..vals)
                .map(|v| {
                    let base = v * HUES * SATS;
                    (curvature_weights(cw, base, shape.curvature, true), curvature_weights(cw, base, shape.curvature, false))
                })
                .collect()
        } else {
            Vec::new()
        };
        let hold = HOLD_TIE * mean;
        let last = vals - 1;
        let free_end = matches!(ending, ValueEnding::Hold | ValueEnding::HoldSaturated);
        // the share of the hold at saturation index s (1: held; 0: pinned) and the pin's weight
        let hold_share = |s: usize| -> f64 {
            if ending == ValueEnding::HoldSaturated {
                let t = ((s as f64 / (SATS - 1) as f64) / HOLD_SAT_FULL).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            } else {
                1.0
            }
        };
        let pin_white = HOLD_SAT_PIN * mean;
        let pull_at = |v: usize| -> f64 {
            if ending == ValueEnding::Fade {
                let e = v as f64 / last as f64;
                pull * (1.0 + FADE_PULL * ((e - 0.5) / 0.5).clamp(0.0, 1.0))
            } else {
                pull
            }
        };
        // the curvature penalty of the second difference centred on plane c: stiffer over the upper half of the value
        // range with `bright_smooth`
        let bright = if shape.bright_smooth.is_finite() { shape.bright_smooth.max(0.0) } else { 0.0 };
        let curve_at = |c: usize| -> f64 {
            let e = c as f64 / last as f64;
            curve * (1.0 + bright * ((e - 0.5) / 0.5).clamp(0.0, 1.0))
        };
        let at = |x: &[[f64; 2]], k: usize, i: usize| x.get(k).map_or(0.0, |p| p[i]);
        let top = if free_end { vals } else { vals - 1 };
        let sweeps = if curved { 400 + 1500 } else { 400 };
        for sweep in 0..sweeps {
            let curved_now = curved && sweep >= 400;
            for v in 1..top {
                for h in 0..HUES {
                    for s in 0..SATS {
                        let k = self.index(v, h, s);
                        let (w, data) = (cw.get(k).copied().unwrap_or(0.0), cv.get(k).copied().unwrap_or([0.0; 2]));
                        let mut nb = vec![self.index(v, (h + 1) % HUES, s), self.index(v, (h + HUES - 1) % HUES, s)];
                        if s > 0 {
                            nb.push(k - 1);
                        }
                        if s + 1 < SATS {
                            nb.push(k + 1);
                        }
                        let mut new = [0.0; 2];
                        for (i, slot) in new.iter_mut().enumerate() {
                            // minimise Σ wt·(α·x + β)² over the terms holding x: x = −Σ wt·α·β / Σ wt·α²
                            let (mut num, mut den) = (data[i], w + pull_at(v));
                            // the hold tie (x_last − x_{last−1})², in the equations of both planes it holds; with
                            // HoldSaturated weighted by saturation, and a pin of the white plane taking over at grey
                            let tie = hold * hold_share(s);
                            if free_end && v == last {
                                num += tie * at(&x, self.index(v - 1, h, s), i);
                                den += tie + pin_white * (1.0 - hold_share(s));
                            } else if free_end && v + 1 == last {
                                num += tie * at(&x, self.index(last, h, s), i);
                                den += tie;
                            }
                            for j in &nb {
                                num += kappa * at(&x, *j, i);
                                den += kappa;
                            }
                            if let (true, Some((wh, ws))) = (curved_now, plane_w.get(v)) {
                                let (cn, cd) = curvature_terms(&x, v * HUES * SATS, h, s, i, wh, ws);
                                num += cn;
                                den += cd;
                            }
                            // second differences along value centred at c = v−1, v, v+1 (each needs both neighbours)
                            for c in [v - 1, v, v + 1] {
                                if c == 0 || c + 1 >= vals {
                                    continue;
                                }
                                let (lo, mid, hi) = (self.index(c - 1, h, s), self.index(c, h, s), self.index(c + 1, h, s));
                                let (alpha, beta) = if c == v {
                                    (-2.0, at(&x, lo, i) + at(&x, hi, i))
                                } else if c + 1 == v {
                                    (1.0, at(&x, lo, i) - 2.0 * at(&x, mid, i))
                                } else {
                                    (1.0, -2.0 * at(&x, mid, i) + at(&x, hi, i))
                                };
                                num -= curve_at(c) * alpha * beta;
                                den += curve_at(c) * alpha * alpha;
                            }
                            *slot = num / den;
                        }
                        if let Some(p) = x.get_mut(k) {
                            *p = new;
                        }
                    }
                }
            }
        }
        if shape.support_fade {
            self.fade_by_support(&mut x, total);
        }
        x
    }

    /// [`ValueShape::support_fade`]: as [`Splat::solve`] fades the single-step correction, fade each cell's
    /// brightness-dependent part (toward the single-step correction) by the sample weight near it: within
    /// [`SUPPORT_REACH`] hue and saturation steps on its own value plane and the planes on either side. The part counts
    /// fully once that is clearly above [`SUPPORT_SHARE`] of all the sample weight. Bright near-neutrals (clouds) have
    /// almost no samples, so a correction smoothed into them from elsewhere fades out.
    fn fade_by_support(&self, x: &mut [[f64; 2]], total: f64) {
        let w0 = SUPPORT_SHARE * total;
        let vals = self.vals;
        for v in 0..vals {
            for h in 0..HUES {
                for s in 0..SATS {
                    let mut near = 0.0;
                    for vv in v.saturating_sub(1)..=(v + 1).min(vals - 1) {
                        for dh in 0..=2 * SUPPORT_REACH {
                            let hh = (h + HUES + dh - SUPPORT_REACH) % HUES;
                            for ds in 0..=2 * SUPPORT_REACH {
                                let Some(s2) = (s + ds).checked_sub(SUPPORT_REACH) else { continue };
                                if s2 < SATS {
                                    near += self.w.get(self.index(vv, hh, s2)).copied().unwrap_or(0.0);
                                }
                            }
                        }
                    }
                    let keep = near * near / (near * near + w0 * w0);
                    if let Some(p) = x.get_mut(self.index(v, h, s)) {
                        p[0] *= keep;
                        p[1] *= keep;
                    }
                }
            }
        }
    }
}

/// `table` (with value divisions) with a correction `corr` of one value plane (`HUES × SATS`, as [`Splat::solve`]
/// gives) composed on every plane: the single-step part of a map with value divisions. The linear value ratio becomes
/// the encoded scale at each plane's value.
pub fn compose_all_planes(table: &HsvTable, corr: &[[f64; 3]]) -> HsvTable {
    let mut t = table.clone();
    let plane = corr.len().max(1);
    let last = (t.val_divisions.max(2) - 1) as f64;
    let encoded = t.srgb_value;
    for (i, e) in t.data.iter_mut().enumerate() {
        let Some(c) = corr.get(i % plane) else { continue };
        let ev = (i / plane) as f64 / last;
        e[0] = (f64::from(e[0]) + c[0]).clamp(-170.0, 170.0) as f32;
        e[1] = (f64::from(e[1]) * c[1].exp()).clamp(0.0, 8.0) as f32;
        let k = if encoded && ev > 0.0 {
            let l = f64::from(lcc_color::transfer::srgb_to_linear(ev as f32));
            f64::from(lcc_color::transfer::linear_to_srgb((l * c[2].exp()) as f32)) / ev
        } else {
            c[2].exp()
        };
        e[2] = (f64::from(e[2]) * k).clamp(0.0, 8.0) as f32;
    }
    t
}

/// `table` with the brightness-dependent deviation `dev` (from [`ValueSplat::solve`]) composed on top.
pub fn compose_value(table: &HsvTable, dev: &[[f64; 2]]) -> HsvTable {
    let mut t = table.clone();
    for (e, d) in t.data.iter_mut().zip(dev) {
        e[0] = (f64::from(e[0]) + d[0]).clamp(-170.0, 170.0) as f32;
        e[1] = (f64::from(e[1]) * d[1].exp()).clamp(0.0, 8.0) as f32;
    }
    t
}

/// How fast a map with value divisions changes along value: the largest hue-shift change (°) and saturation change
/// (|ln ratio|) between neighbouring value planes, per unit of encoded value, over the cells with saturation where hue
/// means something (from the third saturation division up). `(0, 0)` for a single-step map.
#[cfg(test)]
pub fn value_steepness(t: &HsvTable) -> (f64, f64) {
    value_steepness_quantile(t, 1.0)
}

/// [`value_steepness`] at quantile `q` (0–1) of the cells' changes instead of the largest: what most of the map does,
/// without the few steepest cells.
pub fn value_steepness_quantile(t: &HsvTable, q: f64) -> (f64, f64) {
    let (vals, hues, sats) = (t.val_divisions, t.hue_divisions, t.sat_divisions);
    if vals < 2 || t.data.len() < vals * hues * sats {
        return (0.0, 0.0);
    }
    let step = 1.0 / (vals - 1) as f64;
    let (mut hue, mut sat) = (Vec::new(), Vec::new());
    for v in 1..vals {
        for h in 0..hues {
            for s in 2.min(sats)..sats {
                let (Some(a), Some(b)) = (t.data.get(((v - 1) * hues + h) * sats + s), t.data.get((v * hues + h) * sats + s)) else {
                    continue;
                };
                hue.push(f64::from((b[0] - a[0]).abs()) / step);
                if a[1] > 0.0 && b[1] > 0.0 {
                    sat.push(f64::from((b[1] / a[1]).ln().abs()) / step);
                }
            }
        }
    }
    let at = |mut v: Vec<f64>| -> f64 {
        v.retain(|x| x.is_finite());
        v.sort_by(f64::total_cmp);
        let q = if q.is_finite() { q.clamp(0.0, 1.0) } else { 1.0 };
        v.get(((v.len().saturating_sub(1)) as f64 * q).round() as usize).copied().unwrap_or(0.0)
    };
    (at(hue), at(sat))
}

/// The hue groups the per-stop steepness rule checks: oranges/yellows and blues (HSV hue of the map's input, linear
/// ProPhoto), from saturation [`STOP_MIN_SAT`] up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopGroup {
    Warm,
    Blue,
}

/// Lowest saturation a cell needs to count in the per-stop steepness (below it hue hardly shows).
pub const STOP_MIN_SAT: f64 = 0.15;

impl StopGroup {
    fn of(hue: f64) -> Option<StopGroup> {
        if (10.0..60.0).contains(&hue) {
            Some(StopGroup::Warm)
        } else if (190.0..250.0).contains(&hue) {
            Some(StopGroup::Blue)
        } else {
            None
        }
    }
}

/// The changes of a map with value divisions per stop of brightness, cell by cell: for each neighbouring pair of value
/// planes but the one from black (which spans infinitely many stops), the hue-shift change (°) and the saturation change
/// (|ln ratio|) divided by the stops between the planes, `log2(lin(e_v) / lin(e_{v−1}))` on the sRGB-encoded axis.
/// Pushed into `out` as (group, hue change, saturation change); nothing for a map without value divisions.
pub fn changes_per_stop(t: &HsvTable, out: &mut Vec<(StopGroup, f64, f64)>) {
    let (vals, hues, sats) = (t.val_divisions, t.hue_divisions, t.sat_divisions);
    if vals < 3 || sats < 2 || t.data.len() < vals * hues * sats {
        return;
    }
    let lin = |v: usize| f64::from(lcc_color::transfer::srgb_to_linear(v as f32 / (vals - 1) as f32));
    for v in 2..vals {
        let stops = (lin(v) / lin(v - 1)).log2();
        if !(stops.is_finite() && stops > 0.0) {
            continue;
        }
        for h in 0..hues {
            let Some(group) = StopGroup::of(h as f64 * 360.0 / hues as f64) else { continue };
            for s in 0..sats {
                if (s as f64) / ((sats - 1) as f64) < STOP_MIN_SAT {
                    continue;
                }
                let (Some(a), Some(b)) = (t.data.get(((v - 1) * hues + h) * sats + s), t.data.get((v * hues + h) * sats + s)) else {
                    continue;
                };
                let dh = f64::from((b[0] - a[0]).abs()) / stops;
                let ds = if a[1] > 0.0 && b[1] > 0.0 { f64::from((b[1] / a[1]).ln().abs()) / stops } else { 0.0 };
                if dh.is_finite() && ds.is_finite() {
                    out.push((group, dh, ds));
                }
            }
        }
    }
}

/// Linear RGB of HSV `(h°, s, v)` (the inverse of [`hsv`] for v > 0).
pub fn rgb_of_hsv(h: f64, s: f64, v: f64) -> [f32; 3] {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as usize {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [(r + m) as f32, (g + m) as f32, (b + m) as f32]
}

/// Quantile `q` (0–1) of `v` (nearest rank on the sorted values; 0 when empty).
pub fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    v.retain(|x| x.is_finite());
    v.sort_by(f64::total_cmp);
    let q = if q.is_finite() { q.clamp(0.0, 1.0) } else { 1.0 };
    v.get(((v.len().saturating_sub(1)) as f64 * q).round() as usize).copied().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(h: f64, s: f64) -> usize {
        (h / 360.0 * HUES as f64).round() as usize % HUES * SATS + (s * (SATS - 1) as f64).round() as usize
    }

    #[test]
    fn a_stray_sample_does_not_turn_an_empty_region_but_dense_data_keeps_its_correction() {
        let mut sp = Splat::default();
        // dense: oranges around 30° asking for +10°
        for i in 0..2000 {
            let (h, s) = (20.0 + (i % 20) as f64, 0.3 + 0.3 * (i / 20 % 20) as f64 / 19.0);
            sp.add(rgb(h, s, 0.5), rgb(h, s, 0.5), rgb(h + 10.0, s, 0.5), 1.0);
        }
        // one stray saturated purple (0.05 % of the weight, as in a 5-photo Lumix G9 fit) asking for +80°
        sp.add(rgb(270.0, 0.85, 0.5), rgb(270.0, 0.85, 0.5), rgb(350.0, 0.85, 0.5), 1.0);
        let x = sp.solve(0.5);
        let dense = x[cell(30.0, 0.45)][0];
        let stray = x[cell(270.0, 0.85)][0];
        assert!((dense - 10.0).abs() < 1.5, "dense oranges get {dense:.1}°, want about 10°");
        assert!(stray.abs() < 5.0, "an empty purple region turns {stray:.1}° from one sample");
    }

    fn rgb(h: f64, s: f64, v: f64) -> [f32; 3] {
        rgb_of_hsv(h, s, v)
    }

    #[test]
    fn the_support_fade_keeps_dense_brightness_dependence_and_drops_a_stray_near_white_one() {
        // dense: mid-bright oranges paler than the single step; one stray near-white blue asking for much more
        let mut sp = ValueSplat::new(6);
        for i in 0..4000 {
            let (h, s, e) = (20.0 + (i % 20) as f64, 0.3 + 0.3 * (i / 20 % 10) as f64 / 9.0, 0.4 + 0.2 * (i / 200) as f64 / 19.0);
            let c = rgb(h, s, lin(e));
            sp.add(c, c, rgb(h, s * 0.8, lin(e)), 1.0);
        }
        let stray = rgb(215.0, 0.08, lin(0.95));
        sp.add(stray, stray, rgb(215.0, 0.20, lin(0.95)), 1.0);
        let at = |x: &[[f64; 2]], h: f64, s: f64, v: usize| {
            x[(v * HUES + (h / 4.0).round() as usize % HUES) * SATS + (s * (SATS - 1) as f64).round() as usize]
        };
        let plain = sp.solve(ValueShape { ending: ValueEnding::HoldSaturated, ..Default::default() });
        let faded = sp.solve(ValueShape { ending: ValueEnding::HoldSaturated, support_fade: true, ..Default::default() });
        let (d0, d1) = (at(&plain, 30.0, 0.45, 2)[1], at(&faded, 30.0, 0.45, 2)[1]);
        assert!(d0 < -0.1 && (d1 - d0).abs() < 0.02, "dense oranges keep their part: {d0:.3} vs {d1:.3}");
        let (s0, s1) = (at(&plain, 215.0, 0.07, 4)[1].abs(), at(&faded, 215.0, 0.07, 4)[1].abs());
        assert!(s1 < 0.25 * s0.max(1e-9) || s1 < 1e-3, "the stray near-white blue fades: {s0:.4} -> {s1:.4}");
    }

    fn lin(e: f64) -> f64 {
        f64::from(lcc_color::transfer::srgb_to_linear(e as f32))
    }

    /// The saturation change a look with brightness-dependent warm colours wants (ln ratio at encoded value `e`):
    /// warm hues (10–70°) fuller in the darks and paler in the brights, nothing elsewhere.
    fn wanted(h: f64, e: f64) -> f64 {
        if (10.0..70.0).contains(&h) { -0.4 * (e - 0.5) } else { 0.0 }
    }

    /// A single-step map (a small warm hue shift, a value scale off 1) and the map fitted on top of it with 6 value
    /// divisions from samples that want [`wanted`]; the blues get no samples.
    fn fitted() -> (HsvTable, HsvTable) {
        fitted_with(ValueEnding::Pin)
    }

    fn fitted_with(ending: ValueEnding) -> (HsvTable, HsvTable) {
        let mut base = identity();
        for h in 0..HUES {
            for s in 0..SATS {
                let hue = h as f32 * 4.0;
                base.data[h * SATS + s] = [if (10.0..70.0).contains(&hue) { 2.0 } else { 0.0 }, 1.0, 1.02];
            }
        }
        let start = with_value_divisions(&base, 6);
        let mut splat = ValueSplat::new(6);
        for hi in 0..90 {
            let h = hi as f64 * 4.0 + 1.0;
            if !((0.0..120.0).contains(&h) || (290.0..360.0).contains(&h)) {
                continue;
            }
            for si in 1..12 {
                for vi in 1..40 {
                    let (s, e) = (si as f64 / 12.0, vi as f64 / 40.0);
                    let input = rgb(h, s, lin(e));
                    let out = start.apply(input);
                    let (ho, so, vo) = hsv(out).unwrap();
                    let target = rgb(ho, (so * wanted(h, e).exp()).min(1.0), vo);
                    splat.add(input, out, target, 1.0);
                }
            }
        }
        (base, compose_value(&start, &splat.solve(ending)))
    }

    #[test]
    fn changes_per_stop_divide_by_the_stops_between_planes_and_leave_out_the_step_from_black() {
        // 4 planes (encoded 0, 1/3, 2/3, 1); a warm cell (hue 40°, saturation 0.5) whose hue shift grows by 2° per
        // plane and whose saturation scale doubles per plane; a blue cell unchanged
        let mut t = with_value_divisions(&identity(), 4);
        let plane = HUES * SATS;
        let warm = 10 * SATS + 15;
        for v in 0..4 {
            t.data[v * plane + warm][0] = 2.0 * v as f32;
            t.data[v * plane + warm][1] = 2f32.powi(v as i32);
        }
        let mut cells = Vec::new();
        changes_per_stop(&t, &mut cells);
        let stops = |v: usize| (lin(v as f64 / 3.0) / lin((v - 1) as f64 / 3.0)).log2();
        let got: Vec<(f64, f64)> = cells.iter().filter(|c| c.0 == StopGroup::Warm && c.1 > 0.0).map(|c| (c.1, c.2)).collect();
        assert_eq!(got.len(), 2, "two steps (1→2, 2→3); the step from black is left out");
        for (k, v) in [2, 3].into_iter().enumerate() {
            assert!((got[k].0 - 2.0 / stops(v)).abs() < 1e-4 && (got[k].1 - std::f64::consts::LN_2 / stops(v)).abs() < 1e-4, "{got:?}");
        }
        assert!(cells.iter().any(|c| c.0 == StopGroup::Blue) && cells.iter().filter(|c| c.0 == StopGroup::Blue).all(|c| c.1 == 0.0 && c.2 == 0.0));
        assert_eq!(quantile(vec![3.0, 1.0, 2.0], 1.0), 3.0);
        assert_eq!(quantile(Vec::new(), 0.99), 0.0);
        let mut none = Vec::new();
        changes_per_stop(&identity(), &mut none);
        assert!(none.is_empty(), "a single-step map has no steps");
    }

    #[test]
    fn the_low_saturation_fade_keeps_grey_on_the_single_step_and_leaves_saturated_cells_alone() {
        let mut dev = vec![[3.0, 0.2]; HUES * SATS * 4];
        fade_low_saturation(&mut dev, 0.25);
        let at = |s: usize| dev[(2 * HUES + 7) * SATS + s];
        assert_eq!(at(0), [0.0, 0.0], "grey keeps the single-step correction");
        assert!(at(2)[1] > 0.0 && at(2)[1] < 0.2 && at(2)[1] < at(5)[1], "near-grey fades smoothly: {:?} {:?}", at(2), at(5));
        assert_eq!(at(8), [3.0, 0.2], "from saturation 0.25 up the part is whole");
        let mut off = vec![[3.0, 0.2]; HUES * SATS * 4];
        fade_low_saturation(&mut off, 0.0);
        assert!(off.iter().all(|d| *d == [3.0, 0.2]), "0 is off");
    }

    /// A splat of an orange region (hue 20–44°, saturation 0.5–0.8) whose samples want ln saturation +0.1, with one
    /// cell (hue 32°, saturation 20/29) where one photo with `dense` sample weight wants −0.5: the pothole.
    fn pothole_splat(cap: f64, dense: f64) -> (Splat, usize) {
        let mut sp = Splat::with_cap(cap);
        let sample = |sp: &mut Splat, photo: usize, h: f64, s: f64, corr: f64, w: f64| {
            let input = rgb_of_hsv(h, s, 0.5);
            sp.add_from(photo, input, input, rgb_of_hsv(h, (s * f64::exp(corr)).min(1.0), 0.5), w);
        };
        for h in (20..=44).step_by(2) {
            for k in 0..=12 {
                let s = 0.5 + 0.025 * f64::from(k);
                sample(&mut sp, 1 + (h as usize + k as usize) % 8, f64::from(h), s, 0.1, 1.0);
            }
        }
        sample(&mut sp, 99, 32.0, 20.0 / 29.0, -0.5, dense);
        (sp, 8 * SATS + 20)
    }

    #[test]
    fn the_curvature_term_fills_a_pothole_of_conflicting_samples_and_keeps_the_broad_correction() {
        let (sp, k) = pothole_splat(0.0, 3.0);
        let plain = sp.solve_with(0.5, 0.0);
        let curved = sp.solve_with(0.5, 40.0);
        assert!(plain[k][1] < -0.2, "without the term the dense cell stays a pothole: {:?}", plain[k]);
        assert!(curved[k][1] > plain[k][1] + 0.15, "the term pulls it toward its neighbours: {:?} vs {:?}", curved[k], plain[k]);
        // inside the region, away from the pothole (hue 24°, saturation 17/29): the broad +0.1 stays
        let far = 6 * SATS + 17;
        assert!((curved[far][1] - 0.1).abs() < 0.04, "curved {:?} plain {:?}", curved[far], plain[far]);
        // off, solve_with is the plain solve exactly
        assert_eq!(sp.solve(0.5), plain);
    }

    #[test]
    fn the_per_photo_cap_stops_one_photo_deciding_a_cell_alone() {
        let (free, k) = pothole_splat(0.0, 50.0);
        let (capped_sp, _) = pothole_splat(0.25, 50.0);
        let (a, b) = (free.solve_with(0.5, 0.0), capped_sp.solve_with(0.5, 0.0));
        assert!(b[k][1] > a[k][1] + 0.1, "capped, the dense photo no longer decides the cell: {:?} vs {:?}", b[k], a[k]);
        let other = 5 * SATS + 17;
        assert!((b[other][1] - a[other][1]).abs() < 0.03, "a cell no photo dominates keeps its correction: {:?} vs {:?}", b[other], a[other]);
        // the helper: one photo with 100 of 130 cut to at most a quarter of the cell's capped weight
        let mut per = vec![vec![(1usize, 100.0, [100.0]), (2, 10.0, [0.0]), (3, 10.0, [0.0]), (4, 10.0, [0.0])], vec![(5, 1.0, [1.0])]];
        per.resize(HUES * SATS, Vec::new()); // a realistic floor: a quarter of the mean cell weight
        let (w, v) = capped(&per, 0.25);
        let kept1 = v[0][0]; // photo 1's weight kept (its correction is 1 per unit weight)
        assert!(kept1 <= 0.25 * w[0] + 1e-9, "{w:?} {v:?}");
    }

    #[test]
    fn the_blue_pull_removes_sky_blues_at_every_saturation_tapers_smoothly_and_leaves_other_hues_alone() {
        let mut dev = vec![[3.0, 0.2]; HUES * SATS * 4];
        pull_blue(&mut dev);
        // hue index h is 4h degrees
        let at = |h: usize, s: usize| dev[(3 * HUES + h) * SATS + s];
        for s in [3, 15, 29] {
            assert_eq!(at(55, s), [0.0, 0.0], "220° (inside) has no brightness dependence at saturation index {s}");
        }
        assert_eq!(at(30, 10), [3.0, 0.2], "120° (green) untouched");
        assert_eq!(at(10, 10), [3.0, 0.2], "40° (orange) untouched");
        let taper: Vec<f64> = (46..=50).map(|h| at(h, 10)[1]).collect(); // 184°..200°
        assert!(taper.windows(2).all(|w| w[1] <= w[0]), "falls smoothly into the range: {taper:?}");
        assert!(taper[1] > 0.0 && taper[1] < 0.2, "partly pulled within the taper: {taper:?}");
        assert!((blue_pull_weight(185.0)).abs() < 1e-12 && (blue_pull_weight(255.0)).abs() < 1e-12);
        assert!((blue_pull_weight(200.0) - 1.0).abs() < 1e-12 && (blue_pull_weight(240.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bright_smoothing_flattens_the_top_of_the_brightness_dependent_part() {
        let steep_top = |t: &HsvTable| {
            let plane = HUES * SATS;
            let k = 10 * SATS + 15;
            (f64::from(t.data[4 * plane + k][1]) / f64::from(t.data[3 * plane + k][1])).ln().abs()
        };
        let (base, plain) = fitted_with(ValueEnding::Hold);
        let spread = with_value_divisions(&base, 6);
        let mut splat = ValueSplat::new(6);
        for hi in 0..90 {
            let h = hi as f64 * 4.0 + 1.0;
            for si in 1..12 {
                for vi in 1..40 {
                    let (s, e) = (si as f64 / 12.0, vi as f64 / 40.0);
                    let input = rgb(h, s, lin(e));
                    let out = spread.apply(input);
                    let (ho, so, vo) = hsv(out).unwrap();
                    splat.add(input, out, rgb(ho, (so * wanted(h, e).exp()).min(1.0), vo), 1.0);
                }
            }
        }
        let stiff = compose_value(&spread, &splat.solve(ValueShape { ending: ValueEnding::Hold, bright_smooth: 20.0, ..Default::default() }));
        assert!(steep_top(&stiff) <= steep_top(&plain) + 1e-9, "{} vs {}", steep_top(&stiff), steep_top(&plain));
        assert!(stiff.data.iter().all(|e| e.iter().all(|x| x.is_finite())));
    }

    #[test]
    fn the_endings_differ_only_toward_white() {
        let (base, pin) = fitted_with(ValueEnding::Pin);
        let (_, fade) = fitted_with(ValueEnding::Fade);
        let (_, hold) = fitted_with(ValueEnding::Hold);
        let spread = with_value_divisions(&base, 6);
        let plane = HUES * SATS;
        // the saturation deviation of a warm cell (hue 40°, saturation 0.5) on value plane v
        let dev = |t: &HsvTable, v: usize| {
            let k = v * plane + 10 * SATS + 15;
            f64::from((t.data[k][1] / spread.data[k][1]).ln())
        };
        // all three: black keeps the single-step correction, the brights are paler
        for t in [&pin, &fade, &hold] {
            assert!(dev(t, 0).abs() < 1e-6);
            assert!(dev(t, 4) < -0.02, "{}", dev(t, 4));
        }
        assert!(dev(&pin, 5).abs() < 1e-6 && dev(&fade, 5).abs() < 1e-6, "pin and fade return to the single step at white");
        assert!(dev(&fade, 4) > dev(&pin, 4), "fade: a smaller deviation just below white ({} vs {})", dev(&fade, 4), dev(&pin, 4));
        assert!((dev(&hold, 5) - dev(&hold, 4)).abs() < 0.5 * dev(&hold, 4).abs(), "hold: white keeps about the step below");
        // the darks hardly differ between the endings
        assert!((dev(&pin, 1) - dev(&hold, 1)).abs() < 0.03 && (dev(&pin, 1) - dev(&fade, 1)).abs() < 0.03);
        for t in [&fade, &hold] {
            assert!(t.data.iter().all(|e| e.iter().all(|x| x.is_finite())));
        }
    }

    fn hue_sat(t: &HsvTable, c: [f32; 3]) -> (f64, f64) {
        let (h, s, _) = hsv(t.apply(c)).unwrap();
        (h, s)
    }

    #[test]
    fn spreading_a_single_step_map_keeps_its_colours() {
        let (base, _) = fitted();
        let spread = with_value_divisions(&base, 6);
        assert!(spread.srgb_value && spread.val_divisions == 6);
        for h in [5.0, 40.0, 130.0, 220.0, 300.0] {
            for s in [0.1, 0.5, 0.9] {
                for v in [0.01, 0.1, 0.4, 0.9] {
                    let c = rgb(h, s, v);
                    let (a, b) = (base.apply(c), spread.apply(c));
                    for i in 0..3 {
                        assert!((a[i] - b[i]).abs() <= 0.01 * a[i].abs().max(1e-3), "{c:?}: {a:?} vs {b:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_fitted_part_follows_brightness_is_pinned_at_black_and_white_and_stays_out_without_data() {
        let (base, fit) = fitted();
        let spread = with_value_divisions(&base, 6);
        let plane = HUES * SATS;
        for (i, (a, b)) in spread.data.iter().zip(&fit.data).enumerate() {
            let v = i / plane;
            if v == 0 || v == 5 {
                assert_eq!((a[0], a[1]), (b[0], b[1]), "black and white keep the single-step correction");
            }
        }
        let sat = |t: &HsvTable, e: f64| hue_sat(t, rgb(40.0, 0.5, lin(e))).1;
        assert!(sat(&fit, 0.2) > sat(&base, 0.2) + 0.01, "dark oranges fuller: {} vs {}", sat(&fit, 0.2), sat(&base, 0.2));
        assert!(sat(&fit, 0.8) < sat(&base, 0.8) - 0.01, "bright oranges paler: {} vs {}", sat(&fit, 0.8), sat(&base, 0.8));
        // the blues had no samples: they keep the single-step correction
        for e in [0.2, 0.5, 0.8] {
            let c = rgb(220.0, 0.6, lin(e));
            let ((h1, s1), (h2, s2)) = (hue_sat(&base, c), hue_sat(&fit, c));
            assert!((h1 - h2).abs() < 0.2 && (s1 - s2).abs() < 0.005, "blue at {e}: {h1},{s1} vs {h2},{s2}");
        }
    }

    #[test]
    fn skin_tones_come_closer_to_the_target_across_brightness() {
        let (base, fit) = fitted();
        let (mut err_base, mut err_fit) = (0.0, 0.0);
        for e in [0.25, 0.4, 0.6, 0.75] {
            let c = rgb(25.0, 0.4, lin(e));
            let (_, s0) = hue_sat(&base, c);
            let want = (s0 * wanted(25.0, e).exp()).min(1.0);
            err_base += (s0 - want).abs();
            err_fit += (hue_sat(&fit, c).1 - want).abs();
        }
        assert!(err_fit < 0.75 * err_base, "skin saturation error {err_fit} vs single step {err_base}");
    }

    #[test]
    fn an_exposure_sweep_from_minus_2_to_plus_2_ev_changes_colour_smoothly() {
        let (_, fit) = fitted();
        for (h, s) in [(25.0, 0.4), (45.0, 0.8), (60.0, 0.3)] {
            let base = rgb(h, s, 0.18);
            let mut last: Option<(f64, f64)> = None;
            for step in -16..=16 {
                let k = 2f32.powf(step as f32 / 8.0);
                let (ho, so) = hue_sat(&fit, base.map(|v| v * k));
                assert!(ho.is_finite() && so.is_finite());
                if let Some((hl, sl)) = last {
                    assert!((ho - hl).abs() < 0.5, "hue jumps at {step}/8 EV: {hl} → {ho}");
                    assert!((so / sl).ln().abs() < 0.04, "saturation jumps at {step}/8 EV: {sl} → {so}");
                }
                last = Some((ho, so));
            }
        }
    }

    #[test]
    fn a_blue_sky_gradient_has_no_jumps_and_the_map_is_not_steep() {
        let (_, fit) = fitted();
        let mut last: Option<(f64, f64)> = None;
        for i in 0..=200 {
            let t = i as f64 / 200.0;
            // a dusk-to-horizon sky: dark saturated blue at the top, near white at the horizon
            let c = rgb(240.0 - 25.0 * t, 0.75 - 0.6 * t, 0.05 + 0.95 * t);
            let (h, s) = hue_sat(&fit, c);
            if let Some((hl, sl)) = last {
                assert!((h - hl).abs() < 0.6, "hue jumps at {t}: {hl} → {h}");
                assert!((s - sl).abs() < 0.02, "saturation jumps at {t}: {sl} → {s}");
            }
            last = Some((h, s));
        }
        let (hue, sat) = value_steepness(&fit);
        assert!(hue <= 15.0 && sat <= 0.75, "steepness {hue}° / {sat} per unit");
        assert_eq!(value_steepness(&identity()), (0.0, 0.0));
    }

    #[test]
    fn clipped_highlights_keep_the_single_step_colour() {
        let (base, fit) = fitted();
        for c in [[1.0f32, 0.6, 0.2], [1.6, 0.9, 0.3], [1.0, 1.0, 0.4], [3.0, 2.0, 1.0], [1.0, 0.95, 0.9]] {
            let ((h1, s1), (h2, s2)) = (hue_sat(&base, c), hue_sat(&fit, c));
            assert!((h1 - h2).abs() < 1e-3 && (s1 - s2).abs() < 1e-4, "{c:?}: {h1},{s1} vs {h2},{s2}");
            assert!(fit.apply(c).iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn hostile_samples_are_ignored() {
        let mut splat = ValueSplat::new(4);
        let bad = [[f32::NAN, 0.1, 0.1], [f32::INFINITY, 1.0, 0.0], [-1.0, -2.0, -3.0], [0.0; 3], [1e30, 1e-30, 0.0]];
        for a in bad {
            for b in bad {
                splat.add(a, b, a, 1.0);
                splat.add([0.5, 0.2, 0.1], a, b, f64::NAN);
            }
        }
        assert!(splat.solve(ValueEnding::Pin).iter().all(|d| d.iter().all(|v| v.is_finite())));
        assert!(ValueSplat::new(0).solve(ValueEnding::Pin).iter().all(|d| *d == [0.0, 0.0]));
        assert!(value_step(f64::NAN, 6) < 6 && value_step(9.0, 6) == 5 && value_step(-1.0, 0) == 0);
    }
}
