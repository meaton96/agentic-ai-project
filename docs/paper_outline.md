# Paper Outline — Fall 2026 Submission Targets

Working document. Section headers are chosen to match each venue's expected
format; content will be filled in incrementally. Source material: the
`AI Agentic Sandbox Project Directions` vision paper (funding-oriented,
RIT), the existing draft at
[`agentic-ml-classification/docs/ml-classification.tex`](../agentic-ml-classification/docs/ml-classification.tex),
and the three implementations in this repo
([`agentic-ml-classification/`](../agentic-ml-classification/),
[`resource-scheduler/`](../resource-scheduler/),
[`agent-sandbox/`](../agent-sandbox/)).

## 1. Venue comparison

| Venue | Deadline | Paper type | Length | Template | Review |
|---|---|---|---|---|---|
| [SANER 2027](https://conf.researchr.org/track/saner-2027/saner-2027-papers) | Abstract **Sep 21**, paper **Sep 25, 2026** | Research track | 10 pages + 2 refs | IEEE `\documentclass[10pt,conference]{IEEEtran}`, single column | Double-blind |
| [EMSE Special Issue](https://emsejournal.github.io/special_issues/2026_SI_Agentic_SE.html) — "Agentic Software Engineering: The Rise of AI Teammates" | **Sep 28, 2026** (rolling review) | Journal article | No hard limit (Springer EMSE norm ~30–40 pp incl. refs) | Springer EMSE journal template | Single-blind (journal norm) |
| [FSE 2027](https://conf.researchr.org/track/fse-2027/fse-2027-papers) | **Oct 2, 2026** | Research track | 18 pages + 4 refs (initial submission) | ACM `\documentclass[acmsmall,screen,review,anonymous]{acmart}`, single column | Double-anonymous |
| [SEAMS 2027](https://conf.researchr.org/track/seams-2027/seams-2027-research-track) | **Oct 21–23, 2026** | Full research / Short / Industry | Full: 10+2; Short & Industry: 4+1 | Not specified on track page — confirm (SEAMS has historically used ACM format; verify before submission) | Not fully specified — confirm blind policy |
| [ICSE 2027 NIER](https://conf.researchr.org/track/icse-2027/icse-2027-new-ideas-and-emerging-results--nier-) | **Oct 23–24, 2026** | New ideas / short | 4 pages + 1 ref | IEEE `\documentclass[10pt,conference]{IEEEtran}` | Double-anonymous, mandatory **Future Plans** section |
| [ICSE 2027 SEIP](https://conf.researchr.org/track/icse-2027/icse-2027-seip) | **Oct 23–24, 2026** | Industry / practice | 10 pages + 2 refs | IEEE `\documentclass[10pt,conference]{IEEEtran}` | Not anonymous-required |

**Timeline risk worth flagging to the PI directly:** SANER's abstract deadline
is ~4 weeks out, and four of the six deadlines (FSE, SEAMS, NIER, SEIP)
land within a 3-week window in October. Six simultaneous submissions from a
single-author project is a lot — recommend picking a "spine" paper written
first (SANER, since its deadline is earliest and its 10-page IEEE format is
close to the existing `ml-classification.tex` draft already in the repo),
then deriving the others from it rather than writing six papers from
scratch. See §3 for which submissions reuse the most material from which.

## 2. Master outline (long-form spine)

This is the superset outline — EMSE and FSE draw on nearly all of it; the
shorter IEEE venues (SANER, SEAMS, NIER, SEIP) are compressions of it (§3).

1. **Introduction**
   - Agentic AI / LLM-agent adoption context, the reliability gap (agents
     narrate plausible results that can be wrong in exactly the ways ML
     pipelines are wrong: bad splits, leakage, silent metric errors)
   - Framing question: not "can agents run a workflow" but "what is the
     smallest system in which agents can't fool themselves, or us, about
     whether a decision was any good"
   - Contributions list (bulleted, 3–5 items)

2. **Motivation and Problem Statement**
   - Why manufacturing / predictive maintenance as the driving use case
     (downtime cost, safety-criticality, need for auditability)
   - Why LLM-agent MAS specifically (division of labor, specialization)
     vs. a single monolithic agent
   - The core failure mode being designed against: an LLM that is
     probabilistic and fluent enough to narrate an incorrect result
     confidently

3. **Related Work**
   - LLM-powered agentic AI systems generally (survey citations from the
     vision paper's reference list — Acharya et al., Masterman et al.,
     Bandi et al.)
   - Multi-agent systems for manufacturing (Leitão, Barbosa, Bussmann)
   - Agentic harness/orchestration engineering (Lin et al. "Agentic
     harness engineering")
   - Agent-environment research frameworks (Meta ARE — schema-first
     agent/run/event specs, the same pattern this project's Sandbox
     infrastructure independently converged on)
   - Position relative to prior work: **TODO** — this needs an actual
     literature search, not just the vision paper's reference list. The
     vision paper's refs are largely survey/position papers; need
     citations to concrete prior *systems* (AutoML agents, other
     LLM-in-the-loop ML pipelines) to argue novelty precisely.

4. **System Design / Approach**
   4.1 Design principle: agents propose, harness decides (Table: division
       of authority — already drafted in `ml-classification.tex`)
   4.2 The deterministic harness / trust boundary (splitting, metrics
       with bootstrap CIs, leakage detection, sandboxed execution)
   4.3 The agent catalog (intake, feature engineering, profiler,
       modeling, verification, deep-dive — six agents, §5 of
       `PROJECT_OVERVIEW.md`)
   4.4 Recipe templates: the constrained middle ground between "agent
       writes arbitrary code" and "no agent decision at all"
   4.5 Dynamic orchestration: agent catalog + planner + deterministic
       plan validator (the propose-then-verify discipline applied to
       *control flow*, not just content)
   4.6 Safety layers enumerated end-to-end (the 8-layer list in
       `PROJECT_OVERVIEW.md` §7)
   4.7 *(FSE / EMSE only — see §3)* Use Case II comparison: the
       resource-scheduler's direct agent-to-agent (A2A) negotiation
       architecture as a second point in the design space — same
       "agents never assert a fact the deterministic layer didn't
       compute" invariant, but peers message each other directly instead
       of routing through a central orchestrator. Useful as a contrast:
       when is A2A safe (negotiation-shaped subproblems with a
       deterministic constraint gate on receipt) vs. when the
       propose/harness-decide loop is preferred.

5. **Sandbox Infrastructure** *(EMSE / FSE / NIER — the broader platform,
   not just the ML pipeline)*
   5.1 Schema-first foundation: AgentSpec / RunSpec, agents and runs as
       declarative data rather than bespoke orchestration code
   5.2 MCP-mediated tool surface and per-binding access control
   5.3 Observability: append-only, per-run typed event log with
       configurable redaction
   5.4 Isolation boundary (no outbound network access by default —
       structural, not policy)
   5.5 Implementation status — **pull current status from
       `agent-sandbox/`**: M0 (pydantic v2 agent/run/event schemas) and
       M1 (headless single-agent runtime — MCP client, model client,
       agent loop, CLI) are built; multi-agent orchestration, the
       service-layer API, and the cross-institution MCP gateway are not
       yet built. State this honestly as partial implementation, not
       "the Sandbox is done."
   5.6 BYOA (Bring-Your-Own-Agent) and the SWAP_AGENT / ALTER_MODEL /
       ALTER_WORKFLOW operators — the marketplace-style vision, and how
       much of it AgentSpec's declarative binding already structurally
       supports vs. what's aspirational

6. **Evaluation**
   6.1 Use cases: NGAFID aviation predictive maintenance (primary),
       Titanic/Iris (binary/multiclass generalization), resource-
       scheduler synthetic factory-floor scheduling (secondary, if FSE)
   6.2 Research questions *(required framing for EMSE; strongly
       recommended for FSE/SANER)* — draft candidates:
       - RQ1: Does the dynamic (planner-based) orchestrator achieve
         parity with the fixed-sequence baseline on standard runs?
       - RQ2: Can the dynamic orchestrator express control flow the
         fixed sequence structurally cannot (e.g., routing straight to
         deep-dive for an explain-only goal)?
       - RQ3: Does the deterministic trust boundary hold under
         adversarial pressure (prompt injection embedded in training
         data) and under real (non-stubbed) LLM output, not just
         hermetic tests?
   6.3 Method: dataset stats (NGAFID: 22 channels, ~1 Hz, 4.2 GB source,
       streaming/bounded-memory featurization), test methodology (unit
       tests for the harness + stubbed-client integration tests for the
       agent loop + real-LLM evaluation scripts), current test count
       (**verify against repo — README says 87 hermetic tests as of
       last documented count, `ml-classification.tex` draft says 138
       including full-loop integration tests; reconcile before
       submission**)
   6.4 Results: static-vs-dynamic parity and the Iris case where the
       dynamic orchestrator succeeded and the static one exhausted its
       hardcoded candidate budget; the planner formatting-slip bug found
       only by real-LLM evaluation (not the hermetic tests) and its
       narrow fix; the adversarial prompt-injection result
       (`final_test_metrics_present` never asserted true without
       finalize genuinely executing)
   6.5 *(SEAMS framing — see §3)* Recast §6.4 as a self-adaptation
       evaluation: Monitor (RunStateSummary) → Analyze/Plan (planner
       proposal) → re-Analyze (deterministic validator) → Execute, with
       the Iris retry-until-a-candidate-passes behavior as an adaptation
       episode

7. **Discussion**
   - What "agents propose, harness decides" buys you that a more
     autonomous design wouldn't (traceable failure, no path from a
     hallucination to a wrong published result)
   - What it costs (agents can't do things that need real code
     generation — feature engineering is deliberately narrower than the
     term usually means; §5.2 of `PROJECT_OVERVIEW.md`)
   - Generalization evidence: pattern reused successfully across three
     dataset types (tabular, long-format time-series, synthetic
     scheduling) and, differently, in the A2A resource-scheduler design

8. **Threats to Validity / Limitations**
   - Single local/hosted LLM families evaluated so far; results may not
     generalize across model providers
   - Deep-dive attribution is binary-only (occlusion attribution assumes
     a single positive-class probability)
   - Context-window limits on wide feature tables with small local
     models (documented, not yet solved)
   - Single research team / single institution validation — no external
     replication yet
   - Sandbox multi-agent orchestration and MCP governance extensions
     (§4 of the vision paper) are proposed, not yet built — be explicit
     about what's implemented vs. aspirational

9. **Future Work**
   - **Migration from the custom harness/agent-runtime to Amazon
     Strands** — flagged by the PI as coming soon; not yet in the repo.
     Once done, this is strong material for ICSE SEIP specifically (a
     practitioner's account of swapping a ~260-line custom tool-calling
     loop for a managed agent framework while preserving the
     deterministic trust boundary — what had to change, what didn't,
     what broke). For now: state it as planned work and reserve a
     paragraph/section for it once the migration lands.
   - Cross-run evidence reuse (Phase 6, on hold), parallel candidate
     search (Phase 7, on hold)
   - MCP architecture extensions for cross-institution data governance
     (vision paper §4)
   - Emergent cybersecurity threat surface in MAS (vision paper §4) —
     currently framed as future work, not evaluated
   - Agentic marketplace / innovation-lab model for university-industry
     collaboration (vision paper §4)

10. **Conclusion**

**References** (BibTeX — reuse `ml-classification.tex`'s bibliography as a
seed, add the vision paper's 58-entry reference list where relevant, add
the concrete prior-systems citations flagged as TODO in §3 above)

**Data Availability statement** (required by SANER "encouraged", required
by FSE) — decide what's actually shareable: code is in this repo (public?
private?), NGAFID data has its own Kaggle license, RIT API usage may not
be externally reproducible without a partner's own key — draft this
carefully rather than promising more than can be delivered.

## 3. Per-venue section maps

Each row below is what to cut/compress/reframe from the master outline
(§2), not a from-scratch structure.

### SANER 2027 — Research track (10+2 pp, IEEE, earliest deadline)

Closest existing match: `agentic-ml-classification/docs/ml-classification.tex`
is already ~80% of a SANER-shaped paper, just in the wrong LaTeX class
(`elsarticle` instead of `IEEEtran`) and missing Related Work / Threats to
Validity as separate sections. Recommended as the **first paper to
finish**, both for the deadline and because it de-risks the others.

- §1 Introduction (compressed)
- §2 Motivation (compressed, fold into Introduction if tight)
- §3 Related Work (new — currently missing from the .tex draft)
- §4 System Design (4.1–4.6, skip 4.7 A2A comparison — out of scope for a
  focused SANER paper)
- §5 omitted or one short subsection — SANER is empirical-SE-methods
  focused, not platform/infrastructure focused; keep Sandbox context to
  1–2 sentences in the intro, not a section
- §6 Evaluation (6.1–6.4, RQs strongly recommended)
- §7 Discussion (short)
- §8 Threats to Validity (**required-in-spirit** — SANER reviewers expect
  this explicitly, even though the CFP doesn't mandate a named section)
- §9 Future Work (short)
- §10 Conclusion
- Data Availability statement (SANER explicitly calls this out)

### EMSE Special Issue — "Agentic Software Engineering: The Rise of AI Teammates"

Journal-length; use the **full master outline**, §1–10, all subsections.
This is the venue to be most rigorous and complete in: full Related Work
survey, all three RQs with dedicated results subsections, both use cases
(ML pipeline + resource-scheduler) as points of comparison, explicit
"AI teammate" framing in the Introduction and Discussion tying back to the
special issue's own language ("how agents collaborate... what kinds of
collaboration patterns emerge") — worth explicitly discussing the
Verification Agent and Deep-Dive Agent as instances of an "AI teammate"
whose authority is structurally bounded, which is a distinctive angle
relative to typical AIDev-style code-review-agent framing the special
issue's other likely submissions will use. Optional: note in the cover
letter whether/why the AIDev dataset isn't used (this project uses its own
generated run/transcript data instead — a deliberate difference worth one
sentence, not a gap to apologize for).

### FSE 2027 — Research track (18+4 pp, ACM, most room)

Second-most room after EMSE. Use §1–9 including **§4.7 and the
resource-scheduler as a genuine second case study**, not just a mention —
this is the paper where the harness-mediated vs. direct-A2A architectural
contrast can be a real contribution (a design-space comparison, not just
"we also built this other thing"). Include §5 (Sandbox Infrastructure) at
moderate depth since FSE audiences are receptive to systems/infrastructure
contributions. Full RQs, full threats-to-validity.

### SEAMS 2027 — Research track (Full 10+2 / Short 4+1 / Industry 4+1)

Reframe, don't just compress — SEAMS wants self-adaptive systems framing
specifically. Structure:

- Introduction (self-adaptation framing: LLM agents as the adaptation
  logic, deterministic harness as the safety envelope around adaptation
  decisions)
- Problem: adaptive control flow needs to be *trustworthy* adaptation
- Approach: map the dynamic orchestrator directly onto a MAPE-K-style
  loop —
  - **Monitor:** `RunStateSummary` (JSON-safe state, never raw data)
  - **Analyze/Plan:** the planner agent's proposed next action
  - **(new) Re-Analyze:** the deterministic plan validator re-checking
    the proposal against the real catalog/state before anything executes
    — this is the paper's novel framing: an adaptation loop with an
    independent, non-LLM validation stage between Plan and Execute
  - **Execute:** the catalog agent runs
- Evaluation as an adaptation-evaluation: the Iris retry-until-pass
  episode (adaptive policy emerging from a real decision point, not a
  hardcoded count) and the adversarial-injection result (adaptation
  loop's state bookkeeping stays honest even when the environment
  actively tries to manipulate the Analyze stage)
- Decide **Full vs. Short vs. Industry** track once the SANER paper
  exists — the Full track (10+2) has room to reuse most of it with the
  SEAMS reframing; the Industry track (4+1) is a plausible home if the
  Strands migration (§9 future work) lands in time to be a real
  "industrial experience" story instead of a future-work paragraph.
- **TODO before drafting:** the SEAMS track page didn't surface a
  template or blind-review policy — confirm both (likely ACM, likely
  double-blind, but verify against the actual CFP/EasyChair instance
  before formatting).

### ICSE 2027 NIER — New Ideas and Emerging Results (4+1 pp, IEEE)

This is the venue for the **vision-paper material itself**, not the ML
pipeline paper. Draw primarily from the PDF (`AI_Agentic_Sandbox_Project_
Directions`), not `ml-classification.tex`:

- Introduction (the Sandbox vision, 1 paragraph)
- The AI Agentic Sandbox: Goals 1–4 (BYOA, holistic MAS evaluation,
  agentic marketplace, RIT value-adds), condensed from the vision PDF §2
- Sandbox Design (Fig. 1 from the vision PDF, adapted — partners, agent
  store, agentic operations)
- Emerging Result: the predictive-maintenance MSA as a first proof point
  (1 short subsection, pointing to the fuller SANER paper for detail
  rather than repeating it — NIER papers are explicitly allowed/expected
  to reference a longer companion paper)
- Implementation Status (M0/M1 built; what's not) — NIER rewards honesty
  about what's new-idea vs. already-working
- **Future Plans** (mandatory named section per the CFP) — this maps
  directly onto vision PDF §4's Intellectual Merit bullets (enhanced MCP
  architecture, multi-agent harness engineering, MAS cybersecurity,
  system-level MAS innovation, the university-industry innovation-lab
  model) plus the Strands migration and remaining Sandbox milestones
- Conclusion (short)

### ICSE 2027 SEIP — Software Engineering in Practice (10+2 pp, IEEE)

Practice/experience-report framing — "industrially-relevant problems...
lessons learned." Two viable angles, pick one depending on timing:

**Option A (available now):** an experience report on building a
trustworthy agentic harness — structured around the *real bugs found and
fixed* that `PROJECT_OVERVIEW.md` already documents in detail: the Iris
multiclass crash and what generalizing past it actually required, the
`SibSp * Parch` validator bug (a validation rule reusing a heuristic built
for an unrelated purpose), the planner's formatting-slip bug only found
under real-LLM evaluation, the RIT endpoint's 504s under agentic load and
the `--use-local` mitigation. Frame explicitly as "what breaks when you
put LLM agents in front of a system that has to be right, and how we
caught each one" — this is genuinely SEIP-shaped material already sitting
in the repo's own build log.

**Option B (once Strands migration lands):** replace/extend Option A with
a before/after account of migrating off the custom harness onto Amazon
Strands — what had to be preserved (the trust-boundary invariants), what
Strands provided for free, what didn't map cleanly. This is the stronger
SEIP pitch if the timeline allows finishing the migration before Oct 23;
flag this as a scheduling decision for the PI.

## 4. Open items before drafting begins

- [ ] Resolve authorship — vision PDF still says "Authors: to be added";
      need the actual author list + affiliations for every venue (SANER
      and FSE are double-anonymous, so this affects the submission
      version but still needs to be decided for camera-ready)
- [ ] Reconcile the test count discrepancy (87 vs. 138) against the
      current repo state before it goes in any paper
- [ ] Do the actual related-work literature search — current reference
      pool (vision PDF's 58 refs) is mostly surveys/position papers, not
      competing systems; SANER/FSE reviewers will want the latter
- [ ] Confirm SEAMS template and blind-review policy directly from the
      CFP/EasyChair page (not found on the researchr track page)
- [ ] Decide whether resource-scheduler (Use Case II) is ready enough to
      present as real evaluation material or should stay "architecture
      described, evaluation preliminary" — check current implementation/
      test status in `resource-scheduler/` before the FSE and EMSE drafts
      commit to specific numbers
- [ ] Track the Strands migration; its landing date determines whether
      ICSE SEIP gets Option A or Option B (§3)
- [ ] Draft the Data Availability statement's actual answer (public repo?
      private? dataset licensing?) before it's needed for SANER/FSE
