# Transformer communication transport experiments — full_run2 report

Run: `outputs/full_run2`, 2026-08-28. Models `EleutherAI/pythia-70m`, `EleutherAI/pythia-160m`, `gpt2`; all fourteen experiments; seed 0; float64 weight geometry, float32 forward passes; single machine (RTX 4090), ~55 minutes wall, peak process memory 6.9 GiB. `errors.json` is empty — no stage-level exception in any model — and the package test suite passes 13/13. This run includes the full null-and-control battery: five surrogate families with matched random-plane baselines at every algebra stratum, full-pipeline spectrum-preserving triangle surrogates, identity/token-permuted/bootstrap/outlier transport controls, an aggregate Thomas–Wigner arm on pooled states with genuine common support, a 150-triple co-ablation census with grouped splits, task-relevant survival projectors, per-role intervention labels, and an automatic external-anchors table. Every number below is read from the Parquet/NPZ tables in this directory; the machine-generated section summary is `report.md`.

---

## External anchors (gpt2)

The base stage is checked against Wang's published GPT-2 reference values at run time; all eight anchors land.

| Quantity | This run | Wang reference |
|---|---|---|
| Selected head-head edges at FDR q=0.05 | **1,051** | ~1,051 |
| K-composition above / below chance at \|z\|≥2 | 55.1% / 33.7% | ~55% / ~34% |
| Behavioral induction heads sharing one community | **5 of 5** (L5H1, L5H5, L6H9, L7H10, L7H2) | 5 of 5 |
| Top K-composition writer into those heads | **L4H11** | L4H11 |
| Induction-community mean-ablation, gain destroyed | **94.8%** | ~93.8% |
| PosRatio-plane deletion, gain destroyed | 87.1% | ~92.3% |
| Outlier stream coordinates | **[447, 138]** | [447, 138] |
| Neuron wires beyond \|cos\|=0.2 | 139,736 (exact-Beta expectation 13.8) | ~10⁵ (~10¹) |

---

## A. Scalar map

Dense and factored couplings agree to ≤4.7×10⁻¹⁶ on sampled edges in every model (`dense_crosschecks.parquet`); the eighteen edge classes are fully constructed, with complete scans of the mixed classes (up to 1.04×10⁸ candidates) behind bounded two-tail survivor tables and a complete streamed neuron-neuron census.

**Channel census** (head-head, |z|≥2): the same signature in all three models — Q-composition most coupled (55.6/65.3/61.0% above chance for 70m/160m/gpt2), K next (43.3/50.2/55.1%), V anti-coupled (71.7/63.0/65.5% *below* chance). Selected edges: 55 / 939 / 1,051. Signed neuron wires: the bulk of every class is isotropic (robust scale within 4% of 1/√d) with enormous signed tails — 9,671 / 40,889 / 139,736 wires beyond |cos|=0.2 against exact-Beta expectations of ≈314 / 14 / 14.

**Behavioral identification** (`model_summary.json`). Heads are scored at two offsets on the repeated-block corpus and both scores are stored: the induction offset −(B−1) and the duplicate-token offset −B, which select different populations.

| Model | Induction heads (offset −127) | Duplicate-token heads (offset −128) |
|---|---|---|
| pythia-70m | L3H6 .77, L3H5 .74, L3H1 .69, L3H0 .40, L4H7 .33 | L0H3 .92, L2H6, L0H2, L2H5, L0H5 |
| pythia-160m | L4H6 .97, L8H2 .89, L4H10 .87, L5H0 .77, L5H6 .77 | L1H10 .79, L0H7, L3H3, L6H10, L5H5 |
| gpt2 | L5H1 .94, L5H5 .92, L6H9 .91, L7H10 .89, L7H2 .82 | L0H5 .72, L3H0, L0H1, L7H1, L5H10 |

The gpt2 induction list is exactly the literature's five heads. Duplicate-token heads concentrate in early layers (layer-0 heads can match tokens but cannot do induction), and on the Pythias they include the heads that turn out to feed the induction circuit.

**Communities** (`communities.parquet`): Louvain modularity 0.393/0.350/0.283, five communities per model. The induction-overlap communities: 70m a tight 5-head community {L0H3, L1H4, L2H6, L3H0, L3H6}; 160m a 25-head community containing L2H1, the L3 block (including L3H2), and the detected L4/L5/L8 induction heads; gpt2 a 30-head community containing all five behavioral heads, L4H11, and the L9/L10 downstream readers.

Observed: the weight-only map reproduces Wang's census, selection count, writer identification, outlier coordinates, and wire exceedance on gpt2 essentially exactly, transfers with the same channel signature to both Pythias, and the behavioral layer now identifies the textbook induction heads with the duplicate-token population recorded as its own labeled class.

---

## B. Pair operator

**Core and census.** Every retained core reconstructs its C from ‖M̄‖_F to ≤3.3×10⁻¹⁵. OV operators sit near an even symmetric/skew split (medians 0.511/0.519/0.530) with a one-sided symmetric tail; copying scores span ±1 (strongest copier/suppressor: 70m L5H3 +0.996 / L2H6 −0.916; 160m L10H1 +0.9997 / L8H7 −0.998; gpt2 L11H3 +0.999 / L1H11 −0.9997).

**Mean ablation** (roles now labeled per row). Community ablations destroy 31.8% (70m, ΔNLL 0.10), **86.0%** (160m, ΔNLL 0.52), **94.8%** (gpt2, ΔNLL 0.52) of the induction gain. Single corrected-head deletions are individually modest (largest: 70m L3H1 25.5%; gpt2 L5H1 5.4% — the largest single-head effect in its table) — induction is community-borne, not head-borne. The single most load-bearing *individual* heads are the K-feeders, not the induction heads themselves: deleting 70m's **L2H1** alone destroys 81.1%, and 160m's L3H2 60.2%.

**Sign flip.** Every affected Gram and C is bit-identical (recorded change exactly 0.0) while behavior moves hard. With the ratio to same-head deletion damage now guarded (emitted only when the deletion itself destroys ≥1%), the corrected induction heads carry ratios 1.97–8.38 (median ≈2.5), the feeders 1.14 (L2H1, whose flip destroys 92.3%) and 1.52 (L3H2, flip 91.5%), and the extreme case is gpt2's **L4H11**: deletion 1.8%, sign flip **70.8%** — ratio 38. The orientation register, invisible to the scalar map, carries the previous-token→induction K-composition feed. Suppressor flips are as informative: gpt2 L1H11 (copying −0.9997) flips at +0.39 ΔNLL with negligible deletion effect.

**Spectrum flattening** moves outgoing couplings by median 2.0–9.9% while destroying −0.1–0.05% of induction gain — the double dissociation in continuous form. **Gibbs–Hessian typing** calibrates exactly: sign flips carry exactly 0 state-metric energy in all 87 rows, flattening's frame component is ≤1.2×10⁻²⁹ with spectral energy 0.20–0.77, and plane deletions are frame-dominant (median frame 1.11 vs spectral 0.15). The intervention map is three-axis in the data: flattening = spectral, deletion = frame, sign flip = orientation.

**Thomas–Wigner, two arms.** The single-edge arm is the *documented degenerate case*: across all 414 edge×ridge rows, the reader–writer Gram supports have intersection dimension exactly 0 (disjoint rank-64 subspaces), so every row is the ridge-glued union construction; on it the pair law is machine-exact (≤1.1×10⁻¹⁵), the swap control is exactly odd, and the model operators refuse the law by being high-rank — median |φ_model| ≈ π/2, envelope exceeded 4.2–5.2×, dominant plane holding ~3% of compact energy.

The new **aggregate arm** runs the identical extraction on pooled Grams with *genuine common support* (147 layer-pool pairs, median intersection 510–767 of the union; 3 community-pool pairs). The canonical machinery is again exact (pair residual and forward/reverse oddness both 0 to 5×10⁻¹⁶). The model comparison now grades by pooling coherence: indiscriminate layer pools keep |φ_model| ≈ π/2 against tiny predictions (occupancy 8.4–16.5, correlation ≈0), but the *function-coherent community pools* bring the Pythia model operators down to the envelope scale — 70m occupancy **0.65 (inside the envelope)** with residual 0.38 rad, 160m occupancy 1.6 with residual 0.28 rad — while gpt2's community pool stays quarter-turn (occupancy 7.0). On the Pythias, pooling the induction community is the first setting in the campaign where a real model operator's compact angle lands at the two-boost scale the rank-two law predicts.

Observed: the scalar quotient's missing register is orientation — flips up to 38× their deletion damage at zero coupling change — the state-metric typing separates all three axes exactly, and the rank-two transport law, exact on every canonical pair, is matched by real operators only when the pair is pooled over a function-coherent community, and then only on the Pythias.

---

## C. Weight-side connection

**Layer transport, now with its control battery.** The identity control returns collapse exactly 0 (≤1.6×10⁻¹⁴; bookkeeping verified). The decisive control is the token-permuted correspondence: garbage transports fitted on permuted position pairings collapse the span-center variance *at least as strongly* as the real fits —

| collapse fraction | fitted | permuted floor (3 draws) |
|---|---|---|
| 70m K / Q / V | 0.937 / 0.318 / 0.944 | 0.96–0.98 / 0.44–0.84 / 0.96–0.98 |
| 160m K / Q / V | 0.867 / 0.721 / 0.817 | 0.98–0.99 / 0.82–0.93 / 0.96–0.97 |
| gpt2 K / Q / V | 0.977 / 0.440 / 0.773 | 0.998 / 0.90–0.91 / 0.99 |

so span-center collapse is a fit-anything property of covariant re-dressing and is **not by itself evidence of learned frame structure**; the raw span drift of the medians is simply removable by any orthogonal re-dressing. What survives calibration is the compact content of the *fitted* maps themselves: real ‖log Q‖ (median 16–25 rad across layers) exceeds the sequence-bootstrap noise floor (13–19) in 83–92% of layers while sitting far below the token-permuted magnitude (41–50). The fitted rotations are stable beyond refit noise and much smaller than garbage-fit rotations — genuine, modest frame rotation with depth. Selection stability stays the skeptical readout: pooled-vs-stratified Jaccard moves 0.288→0.406 (70m), 0.561→0.569 (160m), 0.572→**0.406** (gpt2, worse) after transport.

**Outlier arm (gpt2).** Projecting out stream coordinates [447, 138] before fitting raises the held-out polar R² at the middle anchor from 0.054 to **0.504** (median over layers): the two outlier coordinates dominate the raw gpt2 fits, and the frame story is much cleaner without them. (The deleted-fit condition numbers are degenerate in the two zeroed coordinates by construction.)

**V-channel triangles versus full-pipeline surrogates** — the section's headline. The census is 161 (160m) / 56 (gpt2) / 1 (70m — its 55-edge graph supports exactly one triple) valid triples on true support dimension 256, with ~49–55% of compact residues at the log branch cut. Twelve full surrogate draws per model rebuild everything — spectrum-preserving random stream frames per factor, map, empirical nulls, selection, communities, triangles (576 surrogate triangles per 12-layer model). Separations of the real medians from the surrogate draw-median distribution:

| statistic | 160m | gpt2 |
|---|---|---|
| compact holonomy / support dim | **−36.6** (0/12 draws below) | **−102.8** (0/12) |
| positive endpoint distance | **−34.1** (0/12) | −1.9 (0/12) |
| symmetric path residual (shape) | **−51.0** (0/12) | +3.8 (12/12) |
| order-reversal difference | **−11.2** (0/12) | **−16.3** (0/12) |
| \|radial residual\| | +1.0 (12/12) | +5.7 (12/12) |

Real triangle transport is *far* more compact-coherent and order-coherent than spectrum-matched random frames in both models — the communities are **not scalar clusters**; their path structure carries genuine partial parallelism. The two models then split: 160m's triangles are also endpoint- and shape-coherent (the full profile of partial parallel transport), while gpt2's sit *above* the null radially and in shape — its long-range direct V edges are genuinely weak relative to composed two-step routes (median log-radial +3.7), a real magnitude asymmetry, not frame noise. The emitted classifications: 160m "below the surrogate distribution in compact, endpoint, order, shape — partial parallel transport with reduced compact residue"; gpt2 "below in compact, order and above in radial, shape". One census note: the span/coupling-matched outside-community control arm returned no extra rows because the 200-triple census already exhausts each model's entire candidate pool (161/56 triples), so the census itself is the matched population; fully-inside-community triples are rare (0–2) because selected V edges seldom lie wholly inside the K/Q-dominated communities.

**Role-complete Q/K loops** (1,449/9/504 valid): compact class scores ~64–66 under all three bridge families with bridge sensitivity 0.21–0.85 (≈1% of the score) and role-complete ≈ role-collapsed — the construction is well-posed and bridge choice is not the uncertainty, but the loops are dominated by the conditioning of the pseudo-inverted typed maps rather than by a clean holonomy signal.

Observed: with controls in place, the span-drift story inverts — median collapse is a fit-anything floor effect while the calibrated finding is a modest, bootstrap-stable frame rotation — and the triangle surrogates upgrade the communities from "scalar clusters?" to measured partial parallel transport, with compact coherence at −37 to −103 robust sigma against refitted spectrum-matched nulls.

---

## D. Positional geometry

**RW-PCA.** PosRatio bands [9,10] / [7,11] / [0,12] with position-to-token ratios up to 190×, sitting in near-degenerate spectral neighborhoods (nearest log-gaps 0.017–0.061). Across the four factor gauges the selected planes are nearly mutually orthogonal (minimum principal cosine 0.034; the balanced gauge selects different bands entirely), the joined support has full rank 8, and bootstrap refits match the trained plane at median minimum cosine only 0.262/0.110/0.010. Deletions: every named plane is catastrophic (posratio 117.7%/29.6%/87.1% of induction gain at ΔNLL 10.8/9.8/5.5 nats; the 8-dim joined support 125.9%/92.7%/101.3%) while matched random planes do nothing (≤2.3%, ΔNLL ≤0.11). The positional object is a gauge orbit over a nearly degenerate, heavily load-bearing face — not a canonical two-plane.

**RoPE.** Exact loops close at 5.0×10⁻¹⁶ (the flat control; phase-shuffled equally flat); 8 allocated planes, 8 distinct characters, finite-window dictionary rank 14 of 16, 48 of 64 head coordinates pass through untouched. The visibility readout is spectrally truncated at declared energy thresholds: the rotary-restricted factor stack concentrates half its energy in 31 (70m) / 53 (160m) stream directions, 90% in 202/383. Against those truncated visible spans, the dominant positional direction has cosine 0.679 (70m) / 0.476 (160m) and the PosRatio plane minimum cosine 0.823/0.687 — the positional face leans into the rotary-visible span but is not contained in it (minimum cosine 0.42–0.47 at 99% positional energy). The activation-realized positional rank is 127 and flagged corpus-limited (= window − 1), so its saturation is a statement about the 128-position window, not the model. Learned role-complete residual loops, computed with the model's split-half rotary pairing, carry large content in every channel (compact ~61–63, positive-log 52–72, dilation 44–53, shear 28–62); because these magnitudes inherit the Q/K-loop conditioning of section C, the learned-versus-flat decomposition remains classified *unresolved — the loop construction dominates the readout*, which is itself the recorded outcome for this instrument.

Observed: the flat RoPE torus is exact and the positional face is real, load-bearing, gauge-orbital, and only partially inside the rotary-visible span, while the learned positional residual stays unresolved for the recorded instrumental reason.

---

## E. Algebra identification

The identification now runs against its full null suite: surrogate refits at **every** learned (ambient m, generator k) stratum for five families, plus a matched random-k-plane baseline per stratum with the headline reported as **excess closure** (baseline − learned).

The answer: closure is almost entirely codimension. The largest excess closure anywhere is 0.056 (gpt2, m=12, k=3), 0.054 (160m, m=12, k=21), 0.044 (70m, m=8, k=3); at the Albert-relevant m=26 the excess is ≤0.03 everywhere. The family decomposition of that small excess is the informative part: it is destroyed by spectrum-preserving *rotation* of the community's own operators (separations −2.5 to −16.3) and by activation shuffling, while layer-matched *random communities* show closure similar to the real one (separations −0.7 to −1.9 on the Pythias; gpt2's −15.8 at its tiny (12,3) stratum is the one membership-sensitive cell). Per-head sign randomization changes nothing at 1.4×10⁻¹⁵ — the instrument is analytically sign-blind, matching the sign-flip story. So the near-closure is a **stream-frame alignment signature carried by head geometry broadly, not a distinguished subalgebra of a special community** — no candidate fits (best chordal distance at m=26: 0.85/1.47/1.49, all near the random plateau), the exact dimension-52 impostors and compact F4 all close at 0 (identification can never lean on closure), corrupted controls move continuously, and the random-plane baseline is overlaid on the closure figure so the saturation slope reads as what it is.

Observed: against matched random planes and four refitted surrogate families, the community generators show at most 0.056 excess closure that vanishes under spectrum-preserving frame rotation — an alignment signature, not a subalgebra, with no classical or exceptional candidate anywhere near.

---

## F. Causal higher-order structure

**Co-ablation synergy at census scale** (150/56/1 triples; gpt2's 56 is its entire candidate pool). Third-order inclusion-exclusion synergies are near zero for most triples (gpt2 max |0.083| nats) — with one real structure: 160m's copy-circuit backbone is strongly *redundant*. The triple **L1H10 → L2H6 → L3H2** (duplicate-token head → mid feeder → previous-token feeder) has synergy **−1.50 nats** (per-prompt SE ≈ 0.18, ≈ −8 SE) on a triple-ablation effect of 15.1 of the 18.3-nat gain, with neighboring triples at −0.67 and −0.58: the circuit's front end is over-provisioned, and joint deletion destroys much less than the sum of its parts predicts. The prediction arm now has real sample sizes and delivers a clean negative with a quantified leakage bound: best held-out R² for induction synergy is 0.206 ungrouped vs **0.040 grouped** (160m — the difference is the head-leakage the grouped split removes), and adding the triangle features *lowers* every score (e.g. 160m grouped 0.040→0.012; gpt2 grouped −0.355→−0.384) while the synthetic norm-matched positive control confirms the estimator stack would see a ternary signal if present (0.44→0.9996). Outcome family 8 lands: co-ablation nonadditivity is dominated by a few large pairwise-style redundancies; triangle features add no incremental causal prediction at this scale.

**Potential, realized, surviving.** Weight potential predicts realized traffic within channel (Pearson 0.44–0.81; 70m K 0.10 at n=17); V-channel realized magnitudes sit at the potential scale while K/Q realized reads run ~10× below it, with the both-sides term the same order as the one-sided K/Q reads. The survival experiment now carries both projectors: whole-stream perturbations *amplify* through downstream mixing (final/initial median 5.3–6.4), but the **task-relevant component — the projection onto each position's next-token unembedding direction — survives at 0.375–0.80** of the reader-output norm: an order-one task-relevant footprint riding on a 6× norm amplification, i.e. most of the amplification is off-target. The behavioral patches identify both extremes by name: removing one hop of writer contribution destroys 83.3% of 70m's induction gain (L2H1→L3H{1,5,6}, K) and 61.6%/39.2% of 160m's (L3H2→L4H6/L5H0), while gpt2's three strongest long-span potentials (L1H8→L10H9 K/Q at C≈0.235, L1H10→L11H8 V at C=0.289) are behaviorally dormant (|effect| ≤ 0.11%).

Observed: the corrected circuit reads coherently end to end — duplicate-token and previous-token feeders K-feed the induction heads, that front end is measurably redundant (−1.5 nats synergy), single-hop patches on it destroy 39–83% of the gain, task-relevant survival is order-one against a 6× off-target norm amplification, and strong-potential dormant edges exist and are named.

---

## G. Exceptional branch

Synthetic controls only — no model-derived 26-space and marked product were supplied, so no claim about Albert structure in a transformer is made in either direction, and nothing in sections A–F produced a candidate carrier that would open the branch (section E in particular closed against it for this run). Control health is exact: Π₅₂ projector diagnostics ≤2.1×10⁻¹⁵ at ranks 52/273; random wedges at mean 52-fraction 0.169±0.030 against the analytic 52/325 = 0.16, with the basis-free trace formula matching the projector to 8.3×10⁻¹⁷; both defect-partition conventions reconstruct the identity to 5×10⁻¹⁷ and stay separate; F4-preserving rotations leave the cubic invariant to 3.7×10⁻¹⁶ while 273-deformations and full O(26) move it at first order; the hostile ℝ⊕J_spin(25) control separates on every fingerprint (derivation 300 vs 52, Peirce [1,24,2]/[1,0,26] vs [1,16,10], distinct Casimir and generic-eigenvalue patterns).

Observed: the exceptional instrument verifies exactly on synthetic Albert, separates the hostile spin control on every declared fingerprint, and remains closed because the carrier-agnostic program supplied no candidate carrier.

---

## Outcome scorecard (design §25)

1. **Scalar quotient** — causally incomplete; orientation is the missing register (flips to 38× deletion damage at zero coupling change), typed exactly by the three-axis Gibbs map.
2. **Local transport law** — exact as pair geometry in both arms; matched by real operators only on function-coherent community pools (Pythia occupancy 0.65–1.6), refused at single-edge and indiscriminate-pool granularity.
3. **Layer drift** — median collapse is a fit-anything floor effect (permuted floor ≥ fitted everywhere); the calibrated finding is modest bootstrap-stable frame rotation; gpt2's outlier coordinates dominate raw fits (R² 0.054→0.504 without them).
4. **Path memory** — answered: communities are not scalar clusters; compact and order coherence beat full-pipeline spectrum-matched surrogates at −11 to −103 robust sigma; 160m adds endpoint/shape coherence, gpt2 adds a real radial (magnitude) asymmetry.
5. **Positional support** — gauge orbit over a nearly degenerate, load-bearing face; not a stable two-plane; partially inside the truncated rotary-visible span.
6. **Algebra identity** — no near-closed subalgebra: excess closure ≤0.056 over matched random planes, frame-borne (killed by spectrum-preserving rotation), sign-blind, no candidate fits.
7. **RoPE residual** — flat part exact; learned residual unresolved because the loop construction dominates (recorded instrumental outcome).
8. **Ternary causality** — answered negative with a validated instrument: one large redundancy structure (−1.5 nats on 160m's copy backbone), no incremental triangle-feature prediction, grouped-vs-ungrouped leakage quantified (0.206 vs 0.040).
9. **Realized traffic** — potential predicts realized within channel (r 0.44–0.81); task-relevant survival order-one against 6× off-target norm amplification; dormant giants named.
10. **Exceptional structure** — instrument healthy, branch closed for lack of a carrier; section E argues against supplying one from this data.

## Method notes and remaining smaller gaps

The surrogate/control arms added in this run: five algebra surrogate families (layer-matched random communities ×100, spectrum-preserving rotations ×25, permuted head identities ×25, sign randomization ×8 as an analytic invariance check, activation shuffles ×25) with random-k-plane baselines ×100 at every learned stratum; 12 full-pipeline triangle surrogate refits per model (map → nulls → selection → communities → triangles); transport controls (identity, 3 token-permuted correspondence draws with full covariant recomputation, 24 random-partition and 24 within-sequence bootstrap refits, outlier-deleted fits); the aggregate Thomas–Wigner arm (layer pools and community pools through the identical extraction, with the single-edge arm retained as the documented zero-intersection degenerate case); grouped-and-ungrouped synergy splits at census scale; and per-position next-token unembedding survival projectors.

Still open, in rough order of value: the BCH cross-check plane for the Thomas–Wigner extraction; a sampled joint-O(d_h)-rotation invariance test for the gauge ledger; the Gibbs spectral-contrast matrix a_ij and the gauge-specific/finite-temperature deletion arms in RW-PCA; an isometrization intervention; per-stratum candidate-fit baselines for the algebra menu; and denser fully-inside-community V-triangle coverage, which would need selection loosened on the V channel inside communities (the current census shows such triples are intrinsically rare, which is itself a finding about where the V-graph lives relative to the K/Q communities).
