# Ball–bat / ball–racket contact models: literature review and options for the X2 tennis-ball task

Research report. No source, test or doc file was modified. Repository claims were read from the cited files. Literature claims carry a URL. **Inference** marks my own reasoning, never a source statement.

## 1. Summary

Both literatures describe the collision as a lossy spring — a damped, nonlinear ball spring in series with a compliant implement spring — lasting about 1 ms for baseball and about 5 ms for a tennis ball on strings. The coefficient of restitution (COR) is a joint property of the pair, not of the ball, and it falls as impact speed rises in every measured dataset found here, in both sports. Energy loss is dominated by hysteresis inside the ball; implement compliance matters only when it returns energy on the contact timescale (hollow-bat trampoline, tennis string bed). For our model the most important gap is not the normal direction: the tangential and rotational side of contact is entirely absent (no rolling resistance, no rolling-friction dimension, solid-sphere inertia), while slow rolling at centimetres per second dominates a balancing task. The normal direction is calibrated only loosely — a ±30 mm test tolerance at the smallest drop admits COR anywhere in about 0.84–1.00 — and on a timescale roughly twice too long.

## 2. What the source says

Source set: D. A. Russell's Penn State/Kettering pages and two co-authored papers hosted there. Quotes are verbatim.

### 2.1 COR: definition, measurement, constancy

- "The coefficient of restitution (COR) is the negative of the ratio of the ball's velocity after impact to its velocity before impact." — https://www.acs.psu.edu/drussell/bats/ball-bat-1.html
- "For a two-body collision, the COR is defined as the ratio of the relative velocity after the collision to that before the collision." / "The COR is actually a joint property of the two colliding bodies." — https://www.acs.psu.edu/drussell/bats/Papers/Nathan_Trampoline-ISEA2004.pdf
- Measurement method: "The COR of the balls was measured by taking the ratio of the rebound speed to the incoming speed", with balls fired from a cannon at a barrel "fixed at each end so that is could not move, though it was able to compress during the impact." — https://www.acs.psu.edu/drussell/bats/nonlinear-ball.html
- Speed dependence, listed as a *false* prediction of the linear model: "COR does not depend on initial velocity of the ball => Laboratory data shows that COR decreases roughly linearly with increasing impact velocity." — ball-bat-1.html. Also: "Research has shown that the COR roughly decreases linearly with increasing ball velocity." — https://www.acs.psu.edu/drussell/bats/ball-bat-0.html

**Source interpretation:** COR is neither constant nor purely a ball property; the ball-alone value `e0` is defined separately as the COR against a massive rigid wall. **Inference:** a single fitted COR cannot hold across a wide speed range, but over a narrow band such as our 1.9–3.1 m/s fit the error may be smaller than the fit's own tolerance.

### 2.2 Impact speed versus COR / energy loss

- "For a baseball or softball at speeds typical of the game, e0 is about 0.5, so that 75% of the initial energy stored in the ball is dissipated." — trampoline paper.
- "Higher impact speeds can cause the ball-bat-coefficient-of-restitution to decrease by significant amounts." — https://www.acs.psu.edu/drussell/bats/bbs-asa.html
- **No numeric table exists.** The COR-versus-speed data for three baseballs and a softball appear only as an image (`bats/images/cor-bat.gif`) on nonlinear-ball.html; the page gives the trend and method but no values and no fit coefficients. The GIF could not be decoded by the fetch tool, so no figure in this report is read off it.
- Contact time also falls: "the duration of the contact time between the ball and the object it strikes also decreases with increasing ball speed." — nonlinear-ball.html

### 2.3 Energy-loss mechanisms

- "most of the initial center-of-mass kinetic energy is converted into compressional energy in the ball, and about 75% of that energy is dissipated." — trampoline paper.
- "When a ball hits a wood bat, it compresses to nearly half its original diameter, losing up to 75% of its initial energy to internal friction forces during this compression." — https://www.acs.psu.edu/drussell/bats/alumwood.html
- "the area contained inside the hysteresis curve represents the amount of energy that is lost to internal friction forces." — nonlinear-ball.html
- Bat vibration is a *loss* for wood: "For solid wood bats, the energy stored in the bat is not effectively returned to the ball but instead appears as low-frequency bending vibrations. Therefore, e never exceeds e0". — trampoline paper.
- Hoop modes are the exception: "the energy stored in the hoop modes is efficiently returned to the ball, resulting in a COR which is larger than e0." — trampoline paper.

**Source interpretation:** the ball dominates the loss; the implement is either a net dissipater (wood, via bending) or a partial energy-return path (hollow barrel, via hoop modes).

### 2.4 Contact time

- "The contact time between bat and is about 0.7 milliseconds (0.0007 s)." (worked 90/110 mph example) — https://www.acs.psu.edu/drussell/bats/impulse.html
- "peak forces in the thousands of pounds required to reverse the direction of the ball in a time of order 1 ms." — trampoline paper.
- "the short time span of the collision (less than 0.001 seconds for baseball and about 0.0015s for softball)". — https://www.acs.psu.edu/drussell/bats/sweetspot.html
- Depends on speed: "Contact time does not depend on the initial velocity of the ball => for a nonlinear spring, the contact time decreases with increasing velocity" (false prediction of the linear model). — ball-bat-1.html
- Depends on implement stiffness: "as rk is reduced from this limit, the effective spring constant is reduced as the bat spring also compresses, resulting in a somewhat longer collision time." — trampoline paper.

### 2.5 Ball stiffness versus implement compliance

- Linear ball model: "m=145.8 g    s=3306.60 N/cm" — i.e. 3.3×10⁵ N/m. — ball-bat-1.html
- Nonlinearity: "the relationship between force and compression is a power law F = k xᵖ" and "a baseball becomes stiffer the more you try to compress it." — nonlinear-ball.html
- Against a rigid wall, "all of the energy losses come from dissipation in the ball." — trampoline paper.
- The controlling ratio: "One of these is the ratio of spring constants rk = kbat/kball; the other is the product fτ". — trampoline paper.
- Shell geometry: "the thickness of a typical aluminum bat is only about 0.1″" and "the stiffness of a shell for such a deformation scales with (t/R)³." — trampoline paper.
- **No numeric k_bat in N/m is given** anywhere on these pages. The paper works only in the ratio `rk` and in hoop frequency.

**Source interpretation:** for a stiff implement the ball supplies essentially all the compliance and all the loss (`e = e0`); when implement compliance approaches the ball's (`rk` of order 1) the stored-energy split changes and COR can exceed `e0`.

### 2.6 Trampoline effect and hoop modes: dynamic or quasi-static?

- "The lowest such mode has a frequency in the 1-3 kHz range and is strongly excited during the collision by the local compression of the shell of the bat at the point of impact." — trampoline paper.
- "As long as the collision time is longer than ~ 1/f , no net energy is transferred to the bat." — trampoline paper.
- "The value of rk at which the COR is maximized occurs when fτ ≅ 1" (τ = collision time). — trampoline paper.
- "The ideal bat would have both rk << 1 … and fτ >> 1 … This could be achieved by a bat consisting of a thin, deformable, but nearly massless membrane, for which the COR is nearly unity independent of e0." — trampoline paper.
- Measured hoop frequencies: "Most commercially available metal baseball bats have hoop frequencies between 1900-2000 Hz" and "If the hoop frequency drops below 900 Hz the bat barrel spring becomes too soft and the collision quickly becomes very inefficient." — https://www.acs.psu.edu/drussell/bats/Papers/Russell_HoopFreq-ISEA2004.pdf

**Source interpretation:** the trampoline effect is explicitly **dynamic and borderline** — `fτ` of order 1, with contact ~1 ms against a mode period of 0.33–1 ms. It is neither quasi-static nor a slow structural mode, and "softer is better" fails below about 900 Hz: there is an optimum.

### 2.7 Sweet spot and node effects

- "the location of the nodes may influence the perception of feel in the batter's hands, but they cannot influence performance." — sweetspot.html
- Why: "The ball does not know that the handle exists … because the vibration impulse returning from the handle end of the bat arrives after the ball has already left the bat." — sweetspot.html
- Yet location does change COR in the same group's model and data: the plotted COR runs across 0.40–0.60 over 5–11 inches from the barrel end. — trampoline paper, Fig. 5.
- "there is often a 1.5 to 2 inch variation in the location of the maximum performance between different types of bats." — sweetspot.html

**Contradiction + inference:** global bending nodes (120–600 Hz, period 1.7–8 ms) cannot establish inside a 1 ms contact, but the hoop mode is a *local* shell deformation driven directly by the contact patch. **Inference:** location therefore matters through local compliance, not through global mode shapes. This resolution is mine, not the source's.

## 3. Tennis-specific evidence

Equally primary for our problem: our ball is an ITF-spec tennis ball (57 g, 67 mm diameter) on a real string bed.

### 3.1 ITF standards: authoritative COR and stiffness for our ball

Verbatim test methods from https://www.itftennis.com/media/15639/2026-technical-booklet.pdf:

- Rebound: "The ball is dropped from a height of 254 ± 0.3 cm (100 ± 0.1 inches) (measured from the bottom of the ball) onto a smooth, rigid and horizontal block of high mass and the rebound height is measured".
- Deformation: "The test measures the deformation of a ball under a load of 80.07 ± 0.5 N (18.00 ± 0.1 lbf) after an initial 'contact' load, of 15.57 ± 0.5 N (3.50 ± 0.1 lbf)"; platen speed 200 mm/min, total load 95.64 ± 0.5 N, forward reading after 5 s. Same sequence described at https://www.itftennis.com/media/2100/balls-ball-approval-tests.pdf ("dropping a ball vertically from a height of 254 cm (100 inches)").

Type 2 (Medium) limits, verbatim table, Appendix I of https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf: mass 56.0–59.4 g; diameter 6.54–6.86 cm; **rebound 135–147 cm**; **forward deformation 0.56–0.74 cm**; return deformation 0.80–1.08 cm.

**Inference (my arithmetic):** implied rigid-surface COR `sqrt(h_r/h_d)` = **0.73–0.76** (sqrt(135/254)=0.729, sqrt(147/254)=0.761). Implied drop-test impact speed ≈**7.1 m/s** in vacuum, slightly less with drag. Implied static secant ball stiffness `95.64 N / 0.0065 m` ≈ **1.5×10⁴ N/m**. Return deformation exceeding forward deformation (0.80–1.08 vs 0.56–0.74 cm) is a direct statement that the ball is hysteretic, i.e. dissipative.

**Usable as calibration targets?** Yes, with two caveats. They are *specification windows*, not measurements of our ball — a conforming ball may sit anywhere in the window. And the COR they imply is a ball-on-rigid-surface value: if we keep the bed rigid and lump bed compliance into the contact law (what the current model does), the target for the lumped pair is **not** this number (see 3.3).

### 3.2 Tennis ball COR versus impact speed

- "The coefficient of restitution in a direction perpendicular to the string plane is typically about 0.90 for a low speed bounce near the middle of the strings but it decreases to about 0.85 for a high speed bounce." — https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/24.%20ObliqueImpact.PDF
- "Casolo (1994) found that the COR for a ball incident normally on a clamped racket decreases from about 0.90 at low ball speeds to about 0.80 at an incident speed of 40 m s⁻¹, while Goodwill & Haake (2003) found that the COR decreased from 0.91 at 24 m s⁻¹ to 0.86 at 41 m s⁻¹." — same paper.
- "the COR for an impact of a tennis ball on concrete is e1 ≈ 0.75." — https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/12.%20Unhappy.PDF

**Inference:** our 1.9–3.1 m/s fit region lies *below* every tennis dataset found (all ≥4 m/s). The low-speed end is an extrapolation in the literature as well as in our model.

### 3.3 String-bed stiffness, and which element dominates the series

- "the strings absorb a significant fraction of the impact energy but almost all of this energy is then returned to the ball. The spring constant of the strings is typically about k2 ≈ 4×10⁴ N/m, giving e ≈ 0.88, as observed"; the ball "has a spring constant k1 ≈ 4×10⁴ N m⁻¹ during the compression phase." — Cross, Unhappy, URL above.
- A racket "strung at a tension of about 55 lb (245 N) … giving a string plane stiffness of about 3×10⁴ N m⁻¹." — same paper.
- "the string plane is more deformable than a court surface, having a stiffness about the same as that of the ball." — Cross, Oblique, URL above.
- "both stringbeds are stiffer than the ball above tensions of 40 lb. Nylon is significantly softer than the ball below 45 lb tension." — https://twu.tennis-warehouse.com/learning_center/stringbeds.php
- At speed: "the strings in a tennis racquet are normally woven, are tensioned to about 300 N, the tension rises to about 500 N during a firm impact with the ball, and the strings stretch by about 20 mm in a direction perpendicular to the string plane." — https://twu.tennis-warehouse.com/learning_center/singlestring.php

**Source interpretation:** the reported quantity is a **dynamic effective spring constant in N/m**, of order 10⁴, with ball and bed of comparable stiffness, so the series pair is roughly half either value. Tension dependence is real but weak above ~40 lb. **Inference:** series of 4×10⁴ with 4×10⁴ gives **2×10⁴ N/m**; series of the static 1.5×10⁴ with 3×10⁴ gives **1×10⁴ N/m**. A lumped contact spring belongs at 10⁴, not at our 6×10³.

### 3.4 Contact time on strings

- "The impact duration on concrete is about 4 ms, and on the strings is about 5 ms." — Cross, Unhappy.
- "The impact duration is typically about 5 ms for high speed impacts on a hand-held racket, but the impact duration was extended at low ball speeds and on a head-clamped racket due to the lower ball stiffness and the larger effective mass of the racket." / "The impact duration varied from about 6.9 ms to about 7.2 ms for all incident ball speeds". — Cross, Oblique.
- "For a normal ball and racquet collision the duration is about 5 ms." — TWU stringbeds.

**Inference:** a lumped 2×10⁴ N/m spring with a 57 g ball gives `t = π√(m/k) = 5.3 ms`, reproducing the measured 5 ms. The quoted stiffness and duration are mutually consistent, which raises confidence in both.

### 3.5 Bed energy loss, trampoline behaviour, snap-back

- "The energy loss in the strings is much smaller than the loss in the ball." — Cross, Oblique.
- The bed has the highest COR and the ball the lowest, "the combined ball-stringbed COR is in-between", and across tensions and materials "At most there is a difference of about 0.02-0.03." — TWU stringbeds.
- Normal and tangential are separate problems: "The dynamics of this interaction in a direction perpendicular to the string plane are determined by the coefficient of restitution. In a direction parallel to the string plane, the dynamics depend on the coefficients of sliding (μS) and rolling (μR) friction, and also depend on the COR." — https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/10.%20StringFriction.PDF (abstract only, medium confidence).
- Measured ball–string sliding friction "yielded a time-average value of 0.43 ± 0.02"; and "The coefficient of sliding friction between the ball and the strings is lower than that normally encountered between the ball and a tennis court where μ ranges from about 0.6 on a fast court to about 0.8 on a slow court." — Cross, Oblique.
- Snap-back is a spin effect: "Low friction strings move more freely across each other and can snap back to their original position while the ball is still on the strings, generating additional spin." — TWU single string.

**Source interpretation:** snap-back and string-on-string sliding change tangential momentum and spin; they are not why the normal COR is 0.88 rather than 0.75. The bed itself has a high COR, so it is a poor damper.

### 3.6 Rolling resistance

- "we have measured rolling friction coefficients, μr … as a function of their angular velocities … we have obtained relations of the form μr = krol·ωball", with `krol` = 0.016 s (old ball), 0.025 s (moderately old), 0.348 s (new ball). — https://arxiv.org/html/0809.4823v2
- The source's own caveat: "We have used balls, available in local market. Most likely, those were trainer balls made not following ITF specifications fully."

**Missing evidence:** the fit is `μr = krol·ω − c` with a negative intercept `c` whose value is not in the retrieved text, so **no μr can be computed for our speeds** (ω ≈ 1–10 rad/s). I do not guess one. What the source does establish is qualitative and sufficient for the decision: a real tennis ball has non-zero, speed-dependent rolling resistance, so our measured zero is wrong.

**Inference from the loss mechanism:** rolling loss is the same material hysteresis that makes COR < 1. A ball with COR 0.75–0.88 in normal impact cannot have exactly zero rolling loss. That argues zero is wrong; it does not supply a value.

### 3.7 Ball rotational inertia (found incidentally, directly usable)

- "I_o = 3.41 × 10⁻⁵ kg m² is the moment of inertia of a tennis ball of mass m = 57 g and radius R = 33 mm about an axis through its CM." — Cross, Oblique.

**Inference:** MuJoCo treats a sphere as uniform, `0.4 m R²` = 2.56×10⁻⁵ kg m² at R = 33.5 mm. The measured value is **33% higher** because a tennis ball is a shell; `I/mR²` is 0.55 measured against 0.40 in our model.

## 4. Gap table

"What our model does" was verified in `src/mjlab/asset_zoo/robots/agibot_x2/tennis_constants.py`, `docs/source/x2_tennis_ball_balance.rst` and `tests/test_x2_tennis_ball_task.py`.

| Effect | What the sources say | What our model does | Matters at slow balancing speeds? | Evidence status |
|---|---|---|---|---|
| Normal stiffness / contact time | Series pair of order 10⁴ N/m (Cross: ball 4×10⁴, strings 4×10⁴; ITF static ball ≈1.5×10⁴ by my arithmetic). Contact 5 ms on strings, 4 ms on concrete, 6.9–7.2 ms clamped at low speed | `solref=(-100000,-22)` direct form; ≈6.0 kN/m; 9–10 ms. **≈3× softer, ≈2× longer** (inference) | Unclear. Only sets static penetration: ≈93 µm at 6 kN/m vs ≈28–56 µm at 1–2×10⁴ N/m. Both small against a 3 mm plate and a 67 mm ball | Ball/bed stiffness: in the source (tennis). 6.0 kN/m: measured here per brief; estimator not found in the files read |
| COR magnitude | Ball-on-strings ≈0.88–0.91 at low speed; ball-on-rigid 0.75 (Cross; ITF window implies 0.73–0.76) | 0.79–0.91 over 1.9–3.1 m/s | Yes, but it is the one thing already fitted | In the source (tennis); measured here |
| COR versus speed | Falls with speed: 0.90→0.80 (0→40 m/s), 0.91→0.86 (24→41 m/s); baseball "roughly linear decrease" | Rises slightly with speed. **Trend inverted** | No over our narrow band; yes as a warning against extrapolating toward 0.1 m/s | In the source (both sports); measured here (4 points) |
| Fit strength | — | 4 points, ±30 mm rebound-height tolerance. At the 0.20 m drop that permits COR ≈0.84–1.00 (my arithmetic). The brief's 0.79–0.91 does not reproduce from the stored pairs, which give 0.92/0.88/0.88/0.92 via `sqrt(h_r/h_d)` | Yes: the normal direction is loosely constrained, not calibrated to ±0.01 | Derived here from `MEASURED_REBOUND` and `REBOUND_TOLERANCE` |
| Energy-loss location | Dominated by ball hysteresis (~75% of stored energy); bed loses little | Single lumped damper. Right in aggregate, wrong in attribution | No | In the source (both); interpretation |
| Ball compliance | `F = k xᵖ` hardening plus hysteresis; return deformation > forward deformation (ITF) | Rigid sphere; deformation only as penetration | No for normal force; yes for rolling loss (same hysteresis) | In the source (both); interpretation |
| Bed compliance / trampoline | Bed stretches ~20 mm perpendicular at 4 m/s; optimum at `fτ ≅ 1`; hoop modes 1–3 kHz; collapse below ~900 Hz | Rigid convex plate | At 0.1 m/s a 3×10⁴ N/m bed deflects ≈0.14 mm, at 2 m/s ≈2.8 mm (inference). **Negligible in our regime** | In the source (both); assumed rigid here |
| Global vibration / hoop modes | Cannot establish inside a 1 ms contact; local hoop mode can | Bat rigid, welded into the kinematic tree | No | In the source (both); interpretation |
| Impact-location dependence | COR 0.40–0.60 across a bat barrel; tennis 0.86 at the node, 0.54 at the tip, 0.71 at the throat | Uniform plate | No at cm/s; yes during a 1–2 m/s impact near the frame edge | In the source (both); assumed uniform here |
| Sliding friction | Ball–string μ ≈ 0.43–0.45; ball–court 0.6–0.8 | `friction[0]=0.6` | Yes whenever the ball slides | In the source (tennis); assumed here |
| Torsional / rolling friction | Non-zero rolling resistance; snap-back and spin are real | `condim=3` → no torsional, no rolling. Measured: a rolling ball holds its speed exactly | **Yes — the dominant defect at cm/s** | In the source (tennis, qualitative); measured here |
| Ball rotational inertia | 3.41×10⁻⁵ kg m² (measured) | Solid sphere, 2.56×10⁻⁵ kg m² | Yes: sets rolling acceleration and spin coupling | In the source (tennis); derived here |
| Air drag | Not addressed by either contact source | None | Unclear below 0.3 m/s; probably negligible | Unknown |

## 5. Options, ranked

Evidence status is one of: measured here / in the source / assumed / unknown. MuJoCo specifics are stated so they can be checked against the manual.

**O1 — Add rolling resistance and the rolling-friction dimension (rank 1).**
*Physical change:* a rolling tennis ball loses energy to hysteretic deformation; ours loses none.
*MuJoCo:* raise bed and ball to `condim=5` (the 3 normal/sliding components plus 2 rolling axes); `condim=4` adds torsional only, the wrong axis for a rolling ball. Set `friction[2]` (rolling) on the **bed** geom, because `priority=1` makes the bed's row govern the pair. MuJoCo's rolling coefficient is a length, not a dimensionless μ; a dimensionless rolling-resistance coefficient μ_r corresponds to `friction[2] = μ_r · R` (**inference — confirm units in the manual before setting a value**).
*Cost:* one scalar, one dimension change; no bodies, joints or step change.
*Breaks:* the normal-drop "no lateral drift" assertion should survive; any test asserting a rolling ball keeps its speed will fail, correctly. Small extra solver work per contact.
*Validation:* roll the real ball across the real bed at 0.10/0.20/0.30 m/s, fit the decay, match within measurement scatter.
*Status:* **assumed** for the value; **in the source** only qualitatively.

**O2 — Correct ball rotational inertia (rank 2).**
*Physical change:* a tennis ball is a shell, not a solid sphere.
*MuJoCo:* set `explicitinertial` on the ball body with `diaginertia=(3.41e-5,3.41e-5,3.41e-5)`, `mass=0.057`, `ipos=(0,0,0)` — the same mechanism already used for the racket bodies in `tennis_constants.py`. No geom or joint change.
*Cost:* three numbers, no runtime cost.
*Breaks:* nothing structural. It raises rotational inertia 33% and lowers pure-rolling acceleration on a slope by about 10% (**inference**), so rolling-distance expectations shift.
*Validation:* same rolling test. A level-bed decay isolates O1 (rolling friction); a tilted bed separates O2 (inertia) from O1.
*Status:* **in the source** (measured 3.41×10⁻⁵ kg m²; the ITF size window brackets our 33.5 mm radius).

**O3 — Re-derive the normal contact from ITF + string-bed stiffness (rank 3).**
*Physical change:* stiffer, shorter contact at similar COR: target `k_series` 1–2×10⁴ N/m and 5–7 ms, instead of 6×10³ N/m and 9.5 ms.
*MuJoCo:* keep `solref` in the direct (negative) form and raise `|stiffness|`, or switch to the positive `(timeconst, dampratio)` form where COR is set directly. Keep `solimp`; it shapes the force–penetration curve, not the rebound.
*Cost:* the four drop measurements must be re-fitted — the current fit is invalidated by construction. The step constraint gets **worse**: at 2×10⁴ N/m the ball–bed period is `2π√(m/k) = 10.6 ms`, only ~10 steps per contact period at 1 ms. That proximity to the stability limit is a plausible reason the original fit landed on a soft spring (**inference**).
*Breaks:* the calibrated 4-point fit, the stored `MEASURED_REBOUND` agreement, and possibly the 1 ms step (a 0.5 ms step would worsen the documented 2.9× cost).
*Validation:* the ITF rebound test — a 254 cm drop onto a rigid surface must rebound 135–147 cm. That is a *window*, so the test is "inside the window", not "equals a number".
*Status:* **in the source** for the target; **unknown** whether our ball is at the soft or firm end of the ITF window.

**O4 — Make COR speed-dependent (rank 4).**
*Physical change:* effective restitution falling with impact speed.
*MuJoCo:* no clean native mechanism. (a) A `solimp` shape that stiffens with penetration makes contact time fall with speed as measured but does not by itself make COR fall; (b) a custom passive force or a per-step restitution impulse at separation, computed from pre-impact normal speed.
*Cost:* (b) is a simulator extension, adds state, and must stay deterministic across `num_envs`.
*Breaks:* determinism survives if the impulse is a pure function of state, but any replay or checkpoint comparison changes.
*Validation:* the four drop heights plus new ones at 0.10 m and 1.00 m; a constant-COR law cannot fit six.
*Status:* **in the source** for the trend; **unknown** for our ball at our speeds.

**O5 — Spring-mounted string-bed plate (rank 5).**
*Physical change:* real normal bed compliance instead of hiding it in the contact law — the physically correct decomposition, and the only way to reproduce the measured ~20 mm bed deflection at high speed.
*MuJoCo:* move the bed geom into its own body with a single `slide` joint along the face normal, `springref`/`stiffness` (or a `position` actuator used passively) plus a damper; keep `priority=1`. Cheaper alternative: keep the rigid plate and put the compliance in a soft `equality` weld.
*Cost:* +1 body, +1 DOF, +2 parameters, and a much stiffer system; the plate mass must be small for a 1 ms step, which pushes frequency up further. Real risk of needing a smaller step.
*Breaks:* the bed is no longer massless and rigid, so `measure_bat_carry_state` and the spawn geometry change; the fit becomes joint over (contact, plate) parameters, under-determined from four numbers.
*Validation:* plate deflection versus impact speed against the ~20 mm at 4 m/s figure — a test at speeds we otherwise do not care about.
*Status:* **in the source** for the deflection magnitude; **not justified** for this task.

**O6 — Mass-spring ball (rank 6).**
*Physical change:* explicit ball compression (shell + core), so the ball stores and returns energy instead of the contact doing it.
*MuJoCo:* split the ball into shell and core joined by a 3-DOF `slide`/`ball` joint with stiffness and damping, or one `slide` along the contact normal with a `connect` constraint recentring the shell. `condim=3` stays on the shell.
*Cost:* high. Three more states per ball, a stiff internal spring inside a 1 ms step, and internal damping nearly degenerate with contact damping.
*Breaks:* drop-fit identifiability, and probably the 1 ms step.
*Validation:* the ITF forward/return deformation pair — 95.64 N must give 0.56–0.74 cm forward and 0.80–1.08 cm return. That is a static *hysteresis* test the rigid-ball model cannot attempt.
*Status:* **in the source** for the target; **assumed** that it changes anything at cm/s.

**O7 — Spatial variation across the face (rank 7).**
*Physical change:* COR varies with impact location (0.54 at the tip to 0.86 at the node in tennis; 0.40–0.60 across a bat barrel).
*MuJoCo:* several bed geoms with different `solref`/`priority` (central sweet zone versus rim), partitioned geometrically.
*Cost:* low mechanically, high in parameters — four drops cannot constrain more than one region.
*Breaks:* nothing structural; it multiplies unidentifiable parameters.
*Validation:* drops at the centre, at the throat end and near the frame.
*Status:* **in the source** (strongly); **unknown** whether it matters with the ball at the centre.

**Not recommended at all.** A trampoline/hoop-mode frame model: needs a 1–3 kHz mode and `fτ ≈ 1`, and the bed deflects under a millimetre at cm/s. An explicit restitution impulse *instead of* a compliant contact: removes the penetration and contact-duration physics a resting ball needs, and is the change most likely to break determinism and the 1 ms step. Air drag, until the rolling work is done.

## 6. Recommendation

**Do O1 and O2 first.** They are the only changes where the physics is unambiguous, the implementation is a few numbers, and the current model fails *categorically* rather than by degree: we measured that a rolling ball holds its speed exactly, and a real ball does not. Both act continuously at centimetres per second, the regime the task lives in. O2 is measured (3.41×10⁻⁵ kg m²) and costs nothing; O1 needs one new measurement (level-bed roll decay) but no new model structure.

**Consider O3 second, not first.** The contact is 3× too soft and 2× too slow, and the ITF drop test is an authoritative target for our exact ball. But the fix pushes the contact period toward the 1 ms step, invalidates the existing fit, and at cm/s only moves static penetration from ~90 µm to ~40 µm — nothing a policy can feel. Do it when the normal direction is next touched for another reason, and re-fit rather than patch.

**Do not do O5, O6 or O7 for this task.** The bed compliance and hoop trampoline are high-speed phenomena: the published 20 mm deflection is at 4 m/s, and the same spring predicts well under a millimetre at 0.1 m/s. They add DOFs, shorten the stable step, and buy nothing in a balancing regime.

**Cannot be justified without new measurement:** (1) the rolling-resistance coefficient — no source gives a usable number at 1–10 rad/s for a conforming ball; (2) the COR of *our* ball on *our* bed below 3 m/s — every tennis dataset is ≥4 m/s and the ITF window is a ≈7 m/s drop; (3) whether the 1 ms step survives a physically stiff 10–20 kN/m contact — a stability question only an experiment answers; (4) the claim that the current model is "calibrated to 0.79–0.91", which overstates the ±30 mm test tolerance.

## 7. Contradictions and missing evidence

**Contradictions.** (1) Russell (sweetspot.html) says location cannot change performance because contact is too short for vibrations; Nathan/Russell (trampoline, Fig. 5) plot COR varying with location. Resolved in §2.7 as local versus global compliance — my inference. (2) Our fitted COR rises with speed while every baseball and tennis source says it falls: a real disagreement, not a labelling difference. (3) The brief's "COR 0.79–0.91" does not reproduce from the stored drop/rebound pairs, which give 0.92/0.88/0.88/0.92. Either the brief uses a different estimator (most likely a velocity ratio at separation rather than from apex) or one is wrong; I could not identify the estimator in the files read.

**Missing evidence.** No numeric baseball COR-versus-speed table exists on the Penn State pages. No string-bed stiffness values in text form from TWU — they are plotted only; the N/m figures here are Cross's. No usable rolling-resistance coefficient at our speeds. No tennis COR data below ~4 m/s. The script that measures 6.0 kN/m was not found, so whether it is a secant slope or `dF/dx` at peak is unknown, and the §3.3 comparison carries that uncertainty.

## 8. Sources

**Kept.**
- Russell, index — https://www.acs.psu.edu/drussell/bats.html — page map only.
- Russell, *Modelling the Bat-Ball Collision: Step 1* — https://www.acs.psu.edu/drussell/bats/ball-bat-1.html — COR definition, ball stiffness, false linear-model predictions.
- Russell, *What happens when ball meets bat?* — https://www.acs.psu.edu/drussell/bats/ball-bat-0.html — COR falls with speed.
- Russell, *Nonlinear Models* — https://www.acs.psu.edu/drussell/bats/nonlinear-ball.html — power-law stiffness, hysteresis, COR method.
- Russell, *Forces between bat and ball* — https://www.acs.psu.edu/drussell/bats/impulse.html — 0.7 ms, sin² waveform.
- Russell, *Sweet spot* — https://www.acs.psu.edu/drussell/bats/sweetspot.html — nodes cannot affect performance; contact-time argument.
- Russell, *Aluminium versus wood* — https://www.acs.psu.edu/drussell/bats/alumwood.html — 75% energy loss in ball.
- Russell, *Regulating bat performance* — https://www.acs.psu.edu/drussell/bats/bat-regulate.html — collision efficiency depends on bat and ball.
- Russell, *ASA BBS standard* — https://www.acs.psu.edu/drussell/bats/bbs-asa.html — COR falls with relative speed.
- Nathan, Russell & Smith, *Trampoline effect* — https://www.acs.psu.edu/drussell/bats/Papers/Nathan_Trampoline-ISEA2004.pdf — joint COR, rk and fτ, hoop modes, COR versus location.
- Russell, *Hoop frequency* — https://www.acs.psu.edu/drussell/bats/Papers/Russell_HoopFreq-ISEA2004.pdf — measured hoop frequencies, 900 Hz floor.
- ITF, *Rules of Tennis 2026*, Appendix I — https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf — Type 2 limits.
- ITF, *Approved Balls… 2026* — https://www.itftennis.com/media/15639/2026-technical-booklet.pdf — rebound and deformation methods and loads.
- ITF, *Ball Approval Tests* — https://www.itftennis.com/media/2100/balls-ball-approval-tests.pdf — 254 cm drop.
- Cross, *Happy balls, unhappy balls, tennis balls* — https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/12.%20Unhappy.PDF — k_ball, k_strings, 4/5 ms, COR versus location.
- Cross, *Oblique impact on strings* — https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/24.%20ObliqueImpact.PDF — COR versus speed, equal stiffness, μ 0.43–0.45, I = 3.41×10⁻⁵.
- TWU, *Energy flow ball–stringbed* — https://twu.tennis-warehouse.com/learning_center/stringbeds.php — 5 ms, weak tension sensitivity.
- TWU (Cross), *Impact with a single string* — https://twu.tennis-warehouse.com/learning_center/singlestring.php — 20 mm stretch at 4 m/s, 17 ms, e = 0.81.
- Jain et al., *Rolling friction of lawn tennis balls* — https://arxiv.org/html/0809.4823v2 — functional form only.

**Deprioritized.** Cross, *String friction* (https://www.physics.sydney.edu.au/~cross/PUBLICATIONS/10.%20StringFriction.PDF) — abstract only. ITF *Ball Research* and the TU Wien thesis — lead generation, no claim kept. Tennis.com and The Atlantic — coaching commentary, no claim kept. `bats/images/cor-bat.gif` — undecodable, no values taken.

---

## 9. Reviewer notes (added by the parent agent after verifying the report)

Three corrections, each checked against the model or by measurement.

**(a) `condim=5` in O1 is not a valid MuJoCo value.** MuJoCo documents `condim` as 1, 3, 4 or 6, and mjlab's `CollisionCfg` validates against `{1, 3, 4, 6}`
(`src/mjlab/utils/spec_config.py`). The rolling-friction dimension is reached with
**`condim=6`**, which carries 3 normal/sliding + torsional + 2 rolling. O1 should read `condim=6`.

**(b) O1's coefficient cannot be derived by formula; it must be calibrated by simulation.** Measured here with a ball rolling without slipping at 0.20 m/s on the bed, `condim=6`:

| `friction[2]` | dv/dt (m/s²) | implied τ (N·m) | τ/(N·R·f2) |
|---|---|---|---|
| 0 | −0.00003 | 8.2e-08 | — |
| 1e-4 | −0.00082 | 2.2e-06 | 1.17 |
| 3e-4 | −0.00715 | 1.9e-05 | 3.40 |
| 1e-3 | −0.05593 | 1.5e-04 | 7.98 |
| 3e-3 | −0.26597 | 7.1e-04 | 12.65 |

`τ/(N·R·f2)` is not constant, so `friction[2]` is not a dimensionless rolling-resistance coefficient and the suggested `friction[2] = μ_r·R` conversion does not hold. Also: `condim=6` with `friction[2]=0` is indistinguishable from `condim=3` (−3e-5 m/s²), so the dimension alone changes nothing. Any value must be fitted to a roll-decay measurement in this simulator.

**(c) The "COR contradiction" in §7 is a misreading, not a disagreement.** The stored pairs `{0.20: 0.17, ...}` are the *user's measured* rebound heights, so `sqrt(h_r/h_d)` recovers the **measured** COR (0.92/0.88/0.88/0.92), which is a statement about the physical ball. The reported 0.79–0.91 is the **simulated** COR, measured as a velocity ratio at separation over 0.1–4.4 m/s. Both are correct and they answer different questions. The model's simulated COR at the four measured drops is 0.788 / 0.881 / 0.911 / 0.887.

**(d) The ±30 mm tolerance criticism stands.** Reproduced: at the 0.20 m drop, ±30 mm of rebound admits COR 0.837–1.000, and the simulation sits at 0.788 against a measured 0.922. The existing test pins rebound height, not COR.

**(e) The positive `solref` form cannot express a 5 ms contact at a 1 ms step.** Checked: MuJoCo requires the time constant to be at least twice the timestep, so the shortest legal positive-form contact at 1 ms is ≈12.6 ms — longer than the current 9.7 ms. Attempts below that boundary return a one-step contact with COR values above 1, i.e. energy injection. The direct (negative) form is the right choice here and this option is not available.

**(f) The 6.0 kN/m estimator** was peak normal force over peak penetration (a secant at peak), from a throwaway script. It is 8.0 kN/m at 0.10 m/s impact and 6.0 kN/m from 1 m/s upward, not constant. The 9–10 ms contact is exactly `π·sqrt(m/k)` for that spring, so the two numbers are mutually consistent.
