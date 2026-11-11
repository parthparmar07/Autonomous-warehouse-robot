# Responsible AI and Sustainability Report

**Project 25 — Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent RL**
ATLAS SkillTech University, uGDX School of Technology
Team: Parth Parmar (2410244), Saket Sarvaiya (2408838), Swarnim Jambhrunkar (2409789)

This report covers ethical requirements ETH-1 to ETH-4 from
[03_requirements_specification.md](03_requirements_specification.md). Everything here concerns a
**simulation**. No physical robot was built or tested.

---

## ETH-1 — Labour impact: productivity, displacement, wages and injuries

The requirement is to weigh evidence on both sides, including evidence unfavourable to automation.

**Productivity.** Coordinated fleets reduce travel and waiting. In our sealed N=5 benchmark, every
learned policy completed 99–100% of orders with zero executed collisions. QMIX delivered fastest
of the three learned policies (46.51 ticks vs 50.99 for IQL). Vendors make much larger
facility-level claims (for example Geek+ reports 75% lower facility power use than a manual
warehouse), but those are vendor estimates, not independent measurements.

**Displacement and wages.** The best-known causal estimate is Acemoglu & Restrepo (2020): across US
commuting zones, 1990–2007, one additional industrial robot per thousand workers reduced the
employment-to-population ratio by about 0.2 percentage points and wages by about 0.42%. The effects
were concentrated on workers without a college degree and in routine manual jobs. That study is
about industrial robots in manufacturing, not warehouse AMRs, and measures local rather than
economy-wide effects, but it is the strongest evidence that the gains are not automatically shared.

Gutelius & Theodore (2019) studied US logistics specifically and did **not** predict mass job
loss in the near term. Instead they expected work intensification, algorithmic management and
de-skilling. They note that people of colour make up about 66% of front-line warehouse workers
against 37% of the US labour force, so the costs of these changes fall disproportionately on them.

**Injuries.** Warehousing is already a high-injury industry: the BLS profile for NAICS 493 reports
about 4.8 recordable cases per 100 full-time workers (California reported 6.1 for 2023). Evidence on
whether robots help is mixed:

- The Strategic Organizing Center (a union coalition, so an advocacy source, using OSHA logs)
  reported a serious-injury rate of 7.3 per 100 at Amazon's robotic warehouses vs 5.7 at its
  non-robotic ones in 2021. It attributes the gap to a faster pace of work set by the robots.
- A 2025 *ILR Review* study of Amazon found severe injuries about 40% lower at robotic
  fulfilment centres, but less-severe injuries about 77% higher, linked to repetitive strain.

**What this means for our system.** Our objective rewards throughput and penalises energy,
lateness and conflicts. It has no term for the humans working beside the robots. If it were
deployed, the pace it sets would become the pace people work to. A responsible deployment would
need explicit limits on human-facing work rates, worker involvement in choosing which efficiencies
to pursue, and monitoring of soft-tissue injury rates, not just collisions.

## ETH-2 — Mapping the safety layer to ISO 3691-4

ISO 3691-4 ("Industrial trucks — Safety requirements and verification — Part 4: Driverless
industrial trucks and their systems", second edition 2023) is the relevant standard. We could only
consult secondary summaries, not the normative text, so every mapping below needs checking against
the published standard.

| ISO 3691-4 concept (from secondary summaries) | Our simulator | Gap to production |
|---|---|---|
| Zones: operating, restricted, and confined (added in 2023); people excluded from confined zones | Cell code 2 = worker corridor, which robots may never enter: a strict exclusion zone | Real robots share space with moving people. Our corridors are static, and we model no zones where people and robots mix. |
| Personnel detection with a protective field that grows with speed; must stop before reaching a person | None. The robot never "sees" a person; corridor cells are known in advance from the map | Safety-rated laser scanners to IEC 61496, protective fields verified at commissioning |
| Detection function at Performance Level d (ISO 13849-1) | The mask is ordinary Python, unit-tested but not safety-rated | A certified safety controller and hardware emergency stop; software is not certified to any performance level |
| Test objects: 200 mm × 600 mm upright cylinder; 70 mm × 400 mm lying cylinder | Not applicable | Physical detection tests |
| Reduced speed when detection is muted (often cited as 0.3 m/s) | The grid moves one cell per tick; there is no continuous speed | Speed monitoring and control |

**A simulation demonstration is not certification.** Zero corridor intrusions in simulation shows
that our software logic is consistent. It says nothing about sensor failure, localisation error,
braking distance or people walking out of corridors. Production conformity would additionally
require:

- a hazard analysis;
- safety-rated sensing and controls at the required Performance Level;
- verification of protective fields;
- site zone marking;
- restart procedures;
- conformity assessment under the applicable machinery regulation.

## ETH-3 — The safety layer is independent of the learned policy

The guarantee does not depend on what the neural network learns. There are two non-learned layers,
both deterministic code:

1. **Action mask before selection** — [`get_action_mask`, src/environment/physics.py:249](../src/environment/physics.py#L249)
   sets FORWARD/BACKWARD to illegal whenever the destination is a shelf (code 1) or corridor
   (code 2), or out of bounds. Masked actions get −∞ value before the argmax, both in training and
   in the deployed ONNX actor (`ActorExport` in
   [src/deployment/onnx_policy.py](../src/deployment/onnx_policy.py)).
2. **Physical refusal in the step** — even if a caller ignores the mask and sends a
   corridor-entering action, [`WarehouseEnv.step`](../src/environment/physics.py) refuses the move,
   holds the robot in place and counts a `safety_override`.

A soft reward penalty sits on top of these but is not relied on.

**Verification.**

- **Unit and adversarial tests** — [tests/test_safety_layer.py](../tests/test_safety_layer.py)
  checks both layers, including a 20-seed adversarial policy that always drives toward the
  corridor.
- **Sealed adversarial episodes** — [scripts/evaluate_scalability.py](../scripts/evaluate_scalability.py)
  runs 20 sealed 600-tick episodes at each fleet size with a policy that ignores the mask and
  drives at any adjacent corridor cell. **Result:** 376,539 corridor-entry attempts were blocked
  across 60 episodes (56,134 at N=5, 110,913 at N=10, 209,492 at N=20), with **0 intrusions** and
  0 executed safety violations. The learned policies also recorded 0 safety violations across all
  180 benchmark episodes.
- **CI gate** — [.github/workflows/verify.yml](../.github/workflows/verify.yml) runs the whole
  test suite on every push and pull request, so a safety-test failure blocks the build.

**Production still needs hardware-level safety stops.** A software mask cannot stop a robot whose
localisation is wrong, whose motor controller fails, or whose map is out of date.

Note: `src/safety/action_mask.py` still contains an older five-action mask from before the v6
rewrite. Only its `orientation_delta` helper is used by the running environment; the enforced mask
is the one in `physics.py` above.

## ETH-4 — Energy: routing savings net of the training footprint

The requirement asks for the routing energy QMIX saves over IQL at N=10, net of the measured
training footprint, as a two-sided estimate.

**Measured: routing energy.** Source: `outputs/evaluation/scalability_N5_N10_N20.json`, 20 sealed
seeds, paired by scenario.

| N=10 | Energy per delivery (simulator energy units, EU) |
|---|---|
| IQL | 11.06 [10.67, 11.50] |
| QMIX | 11.29 [10.90, 11.67] |
| QMIX − IQL (paired) | **+0.23 [−0.17, +0.63]** |

QMIX shows **no measurable energy saving** over IQL at N=10. The point estimate is about 2% *more*
energy per delivery, and the 95% interval includes zero. (QMIX does deliver faster: 42.1 vs 56.9
ticks.)

**Measured: training footprint.** Source: `outputs/emissions/emissions.csv`, CodeCarbon,
India grid at about 0.71 kg CO₂/kWh. The four N=10/N=20 training runs (1.8 h of GPU-laptop time)
used **0.047 kWh, about 34 g CO₂e**; the sealed benchmark used another 0.028 kWh.

- **Upper bound.** CodeCarbon measured the whole machine, and up to two runs overlapped, so these
  figures overlap.
- **Estimated, not metered.** Without admin rights CodeCarbon models CPU power instead of reading a
  meter.
- **N=5 not included.** The N=5 models were trained on a different machine and were not measured.

**Converting simulator units to real energy (assumptions, not measurements).** One EU equals five
empty grid moves (0.2 EU per move). We assume 1 m per cell and a 310 kg loaded AMR with
rolling-resistance coefficient 0.015. That gives about 46 J/m at the wheels, our own first-principles
figure; we found no measured AMR drive-energy reference. We multiply it by 1–4× for drivetrain and
acceleration losses, so **1 EU ≈ 0.23–0.91 kJ**.

| Case | Per delivery | Training paid back after |
|---|---|---|
| Point estimate (QMIX +0.23 EU) | QMIX costs 1.4–5.8 × 10⁻⁵ kWh **more** | Never |
| Most favourable end of CI (QMIX −0.17 EU) | QMIX saves 1.1–4.4 × 10⁻⁵ kWh | ~1,100–4,300 deliveries |

**Conclusion.** We cannot claim an energy benefit for QMIX at this scale. The honest reading is that
any routing saving is smaller than our measurement can resolve. The training footprint is small in
absolute terms (tens of grams of CO₂e). But it is a certain cost, while the saving is not shown to
exist. Two things would decide whether the trade-off is positive:
- a longer horizon with more deliveries;
- independent training seeds, to narrow the interval.

---

## Sources

- Acemoglu, D. & Restrepo, P. (2020). Robots and Jobs: Evidence from US Labor Markets. *Journal of Political Economy* 128(6), 2188–2244. https://econpapers.repec.org/RePEc:ucp:jpolec:doi:10.1086/705716
- Gutelius, B. & Theodore, N. (2019). *The Future of Warehouse Work: Technological Change in the U.S. Logistics Industry.* UC Berkeley Labor Center / Working Partnerships USA. https://escholarship.org/content/qt90p6f8rg/qt90p6f8rg.pdf
- US Bureau of Labor Statistics, Industries at a Glance: Warehousing and Storage (NAICS 493). https://www.bls.gov/iag/tgs/iag493.htm
- California Department of Industrial Relations, 2023 nonfatal injury incidence rates, Table 4. https://dir.ca.gov/oprl/Injuries/2023/2023Table4.html
- Strategic Organizing Center (2022) injury-rate report, as covered by OH&S. https://ohsonline.com/articles/2022/04/13/report-shows-injury-rate.aspx ; SOC (2021) *Primed for Pain*. https://thesoc.org/wp-content/uploads/sites/342/PrimedForPain.pdf
- *ILR Review* (2025) study of injuries at Amazon robotic fulfilment centres, as summarised in OnLabor. https://onlabor.org/amazons-approach-to-robotics-is-seriously-injuring-warehouse-workers/
- ISO 3691-4:2023 secondary summaries: Pilz (2023) https://www.pilz.com/en-INT/company/news/articles/238928 ; TÜV Rheinland AGV whitepaper (2021) https://www.tuv.com/content-media-files/master-content/services/industrial-services/pdf/tuv-rheinland-automated-guided-vehicle-whitepaper-2021.pdf
- Geek+ facility energy claim (vendor estimate), Robotics 24/7. https://www.robotics247.com/article/geek_estimates_its_robotics_solutions_saved_140000_tons_of_carbon_emissions
