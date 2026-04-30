Ready for review
Select text to add comments on the plan
Improved Career Strategy: ML/AI in Medicine — Bayern, Industry-Track
Note on format. This isn't a code plan — it's revised career feedback. I'm using this file because it's the only one I can edit while plan mode is on. Skim the bolded items; everything else is supporting detail.
Context
You are submitting your master's thesis in early August 2026 (~3 months from today, 2026-04-28) at TUM. Thesis: fMRI-based prediction of MCI → Alzheimer's dementia conversion. Architecture (currently GNN) is not locked. You want industry roles in medtech / health-tech, primarily Bayern + Munich, optionally remote-Germany. You received generic but reasonable feedback and want it sharper, more current, and tailored to your situation.
Below: what the original feedback got right, what it got wrong or missed, and a concrete 3-month plan.
1. Honest critique of the original feedback
What it got right
Munich + Berlin are the right geographic anchors for medical AI in Germany.
Python/PyTorch, DICOM/NIfTI, HPC, multimodal data are all genuine table-stakes.
"Package your thesis as a clean repo" is correct — but the original is too vague about how.
Salary bands (~€55–70k entry-level) are realistic.
What it gets wrong, given your goals
It frames your GNN work as an unambiguous strength. For industry CV/medical-imaging roles, GNN-on-fMRI is a niche academic technique. Most postings ask for CNN, U-Net (nnU-Net), Vision Transformers, increasingly medical foundation models (MONAI Model Zoo, VISTA3D, BiomedCLIP). You should not drop GNN, but treating it as your headline skill misaligns you with the keyword filters and interview rubrics most Bayern medtech employers use. Pivot: keep GNN as one experiment, add a CNN/ViT baseline, and frame the thesis around the clinical question (prognosis), not the architecture.
It conflates research and industry tracks. DZNE postdoc and Helmholtz Munich are research-track and require/prefer publications — not your stated goal. Listed as "near-perfect matches" they will pull your application energy in the wrong direction.
It misses the biggest 2026 trends. No mention of: medical imaging foundation models (MONAI Model Zoo, VISTA3D, BiomedCLIP, MedSAM); the EU AI Act which became applicable for high-risk medical AI in 2025–2026 and is now interview-relevant; the lecanemab/donanemab era in Alzheimer's, which has triggered a hiring wave on the pharma side (Roche Penzberg, Boehringer, Bayer) for biomarker / imaging ML — directly relevant to your thesis topic.
The Munich/Bayern industry employer list is shallow. Siemens Healthineers and ZEISS aren't Munich proper; ZEISS is in Oberkochen (BW). The actual Bayern-cluster you should target is broader and more concrete (see §5).
"MLOps / cloud — worth learning" is too soft for a 3-month timeline. With your timeline, MLOps is not optional — it's how a fresh master's grad differentiates against PhDs.
No German-application reality. Lebenslauf format, Anschreiben requirements, Werkstudent-to-full-time pipelines, Blue Card visa logistics, and language expectations matter and aren't mentioned.
Survival / time-to-event framing is missed entirely. MCI → AD conversion is fundamentally a time-to-event problem. Industry survival-analysis-with-imaging is under-developed (per recent reviews) — if you frame your thesis this way, you become rare in interviews.
What's outright stale
"Junior ML in Munich ~€67.5k": this is the floor in 2026, not the average. Mid-tier medtech (Brainlab, deepc, Siemens Healthineers) start at €60–75k for masters, and pharma-side ML (Roche, Boehringer) often starts at €70–85k including bonus.
"C++ nice to have, not critical": for Siemens Healthineers core imaging teams C++ is more like expected for the platform/algorithms ladder. For ML-research-engineer ladders, Python-only is fine.
2. Refined Bayern industry market view (April 2026)
The high-leverage observation: there are four industry segments competing for the same medical-ML profile, and your thesis lets you credibly target three of them.
SegmentExamples (Bayern + remote-DE)Fit for your thesis
Radiology AI platforms / startups
deepc (Munich), mediaire (Berlin, hybrid), Quibim, Avelios Medical (Munich), Floy, Smart Reporting
Strong — frame thesis as "imaging biomarker for prognosis"
Big medtech (devices + software)
Siemens Healthineers (Erlangen, ~1.5h from Munich by ICE), Brainlab (Munich), GE HealthCare DE, Philips DE
Good — emphasize MRI handling, deployment, regulatory awareness
Pharma neuro / biomarker AI (growth area post-lecanemab)
Roche Diagnostics (Penzberg, Bayern — new Innovation Center Feb 2026), Boehringer Ingelheim (Ingelheim/Biberach), Bayer (Berlin/Leverkusen), Merck KGaA (Darmstadt)
Excellent — your topic is directly aligned
Research institutes (skip per your preference)
DZNE, Helmholtz Munich, MPI for Psychiatry
Not target
Key insight you should weaponize: the lecanemab (Leqembi, 2023) and donanemab (Kisunla, 2024) approvals turned Alzheimer's prognosis and patient-stratification into a billion-euro problem for pharma. Your thesis question — "when will an MCI patient convert?" — is exactly the patient-selection problem they need solved for trials and post-approval real-world evidence. Frame it that way.
3. Thesis pivot — small, high-leverage changes
You have ~3 months. Don't replan from scratch. Add three things to your existing pipeline:
a) A CNN or ViT baseline alongside the GNN (1–2 weeks)
Use MONAI ([monai.io](https://monai.io/)) — it's the de facto industry framework. Implementing a 3D CNN or a small ViT baseline in MONAI lets you (i) add a credible architecture to the thesis, (ii) put MONAI on your CV honestly, (iii) report a comparison. Recent papers report 3D CNN on ADNI fMRI hitting ~92% AUC — a fair benchmark for your GNN.
This single change shifts your thesis story from "I tried a niche GNN" to "I benchmarked architectural families on a clinical prognosis problem and chose X for reasons Y."
b) Reframe the prediction head as time-to-event / survival (1 week)
Instead of binary "converter vs non-converter," predict time to conversion using a discrete-time survival head (Logistic Hazard / DeepHit / Cox-PH on top of your encoder).
Use the [pycox](https://github.com/havakv/pycox) library or roll your own — both are cheap to add.
Why: industry survival-with-imaging is under-developed (per the Nature/PMC reviews); pharma trial-design teams care deeply about time-to-event; this differentiates you sharply from "another classifier on ADNI."
c) Multimodal fusion: imaging + tabular biomarkers (1 week)
Add CSF or plasma biomarkers (Aβ42/40, p-tau181, NfL — all in ADNI), age, MMSE, APOE genotype as a tabular branch. Concatenate / cross-attend with the imaging embedding.
This mirrors what every Bayern medtech / pharma team is actually building (Roche's NeuroToolKit is exactly biomarker-panel + imaging fusion).
Cost: low. Payoff: every cover letter you write can now end with "my thesis fuses neuroimaging with clinical biomarkers, which is the same data substrate as your NeuroToolKit / radiology-platform pipeline."
Net result: thesis becomes "Multimodal time-to-event prediction of MCI-to-Alzheimer's conversion combining fMRI and clinical biomarkers, comparing GNN, 3D-CNN, and ViT encoders." — a thesis title that maps directly onto industry job descriptions.
4. Skills to fill in the next ~12 weeks (prioritized)
Tier 1 — must-have, ~2 weekends each:
MONAI end-to-end pipeline on at least one task. Bonus: try a MONAI Model Zoo bundle.
Docker containerizing your inference. ~1 weekend.
FastAPI wrapping your model as POST /predict. ~1 weekend.
W&B or MLflow for experiment tracking on your thesis runs. Use it daily — interviewers ask to see dashboards.
EU AI Act + medical-device basics. Read the [EU AI Act high-risk classification summary](https://artificialintelligenceact.eu/) and skim what IEC 62304 / MDR mean (1–2 hours total). Mention in interviews. Free differentiation.
Tier 2 — should-have, can be lighter: 6. nnU-Net — even one segmentation run with it. It's the segmentation default; gets asked. 7. DICOM — you already know NIfTI. Spend an afternoon reading a DICOM with pydicom, understand SOP/Series/Study hierarchy. 8. One cloud — Azure ML or AWS SageMaker. A 4-hour walkthrough is enough to put on your CV.
Tier 3 — nice to have, defer if time-pressed: 9. ONNX export + TensorRT (Siemens Healthineers asks). 10. C++ basics (only if Siemens Healthineers core-platform roles are on your shortlist).
5. Portfolio strategy — quality over quantity
Hiring managers' consensus: 3 well-engineered, deployed projects beats 15 notebooks. Aim for two artifacts.
Artifact 1: Your thesis repo, productionized
Clean README with a 1-paragraph clinical motivation, architecture diagram, results table.
Dockerfile + docker-compose.yml that brings up a FastAPI service.
A /predict endpoint that takes a NIfTI volume and returns a risk score + time-to-event prediction.
A small Streamlit or Gradio demo (linkable to recruiters).
W&B project link with curves and ablations.
Tests (pytest), pre-commit hooks, CI via GitHub Actions running tests on push.
Two-page PDF report for non-technical recruiters.
This single repo, done well, is worth more than the rest of your CV combined for interviews.
Artifact 2 (optional, only if Artifact 1 is solid first)
Either: a MONAI-based brain-region segmentation demo (proves you can use MONAI and deploy it), or a small "MCI risk dashboard" that wraps Artifact 1 in a clinician-facing UI.
6. Bayern + remote-DE target employer list (industry only)
Concrete shortlist to actually apply to. Roughly ranked by fit-to-thesis:
Roche Diagnostics, Penzberg (Bayern, ~50 min from Munich) — new Diagnostics Innovation Center opened Feb 2026; Alzheimer's blood-biomarker focus; NeuroToolKit. Highest topical fit. Look for "Data Scientist Neurology", "Biomarker Data Scientist", "ML Engineer Diagnostics".
Siemens Healthineers, Erlangen (Bayern) — Europe's largest medical imaging AI employer; AI-Rad Companion, Syngo.via. Look for "AI Solution Engineer", "Research Scientist Medical Imaging", "ML Engineer Imaging AI".
deepc, Munich — radiology AI platform (deepcOS); fast-growing; vendor-agnostic AI marketplace for radiologists. Likely smaller team, more end-to-end work.
Brainlab, Munich — neurosurgery / radiotherapy planning; you have the neuro angle. Look for "Software Engineer ML" / "Computer Vision Engineer".
Boehringer Ingelheim, Biberach / Ingelheim (BW, but remote-friendly to DE) — strong CNS / neurodegeneration pipeline; growing AI team.
Bayer / Merck KGaA / Sanofi DE — pharma data-science teams, often remote-DE.
mediaire (Berlin, hybrid-remote) — explicitly does brain-MRI AI for radiologists; small but topically perfect.
Avelios Medical, Munich — clinical software; ML roles emerging.
Quibim / Floy / Smart Reporting — radiology AI startups, mostly remote-friendly.
Helmholtz Munich — research, but listed because they sometimes have engineer (not scientist) roles that are de-facto industry-style. Apply only to engineer titles.
For each: check careers page directly (Glassdoor / LinkedIn lag), set up email alerts, and apply 8–12 weeks before desired start.
7. Application timeline (April 28 → September start)
WindowThesis workJob-search work
May (now → end of month)
Pivot decisions; CNN baseline running; survival head implemented
Lebenslauf v1 + 1-page Anschreiben template; LinkedIn polished; portfolio repo skeleton; first ~10 applications to top-fit roles
June
Multimodal fusion; ablations; first results figures
First-round interviews; broaden applications to ~25; refine pitch from interview feedback
July
Thesis writing; final figures; W&B dashboards public
On-site / final-round interviews; negotiate offers; target signed offer with Sep–Nov start
Early August
Submit thesis
Sign offer; notify employers of submission date
Reality check: in Germany, full-time medical AI roles for masters grads typically have 6–10 week processes. Starting applications in May is correct timing for September starts, late for August starts. If you want an earlier start, apply this week.
8. German-specific application logistics
Lebenslauf: tabular German format, photo at top right (still common in DE despite being optional), reverse-chronological. Don't use a US-style narrative CV.
Anschreiben: required by ~70% of German employers, including most you're targeting. 1 page, addressed to a named hiring manager when possible. Tailor to each role — generic Anschreiben are rejected.
Language: most Bayern medtech/pharma operate in English internally, so B1 German is usually sufficient — but explicitly stating "B1 German, working toward B2" reads better than silence.
Visa / Blue Card: if non-EU, all the employers above sponsor Blue Cards routinely. The salary threshold for Blue Card in 2026 is ~€48,300 baseline / ~€43,759 for shortage occupations (incl. ML/CS), so anything in the €60k+ band qualifies you cleanly.
Werkstudent path: with a 3-month timeline you should aim straight for full-time, but if your thesis defense slips, a Werkstudent at Brainlab / Siemens Healthineers / deepc is a strong bridge — many convert to full-time within 6 months.
Negotiation: in Germany it's normal to negotiate +5–10% from first offer; less normal to play offers off each other aggressively.
9. What to know in interviews (2026-current)
These are the topics where being current makes you stand out from PhD candidates who studied 2022–2024 and didn't refresh:
Medical imaging foundation models: MONAI Model Zoo, VISTA3D (NVIDIA, unified 3D segmentation, 127 classes), BiomedCLIP, MedSAM. Know the names, what they're for, and one limitation each.
EU AI Act: high-risk classification for medical AI systems; conformity assessment; relationship to MDR/IVDR. Two-sentence answers are fine.
Lecanemab/donanemab era: the clinical context that makes Alzheimer's prognostic AI commercially valuable in 2026. Be able to explain in one sentence why patient-stratification suddenly matters.
Plasma biomarkers (p-tau217 in particular): 2024–2026 saw plasma biomarkers become clinically deployable; if you've fused imaging + biomarkers, you can speak to where each is most informative.
nnU-Net's continued dominance: even though foundation models are rising, nnU-Net is still the segmentation baseline you should benchmark against. Hiring managers test for this.
10. The single sentence pitch you should be ready to deliver
"I'm finishing a TUM master's thesis on multimodal time-to-event prediction of MCI-to-Alzheimer's conversion, combining fMRI and clinical biomarkers, with a benchmark of 3D-CNN, ViT, and GNN encoders — built end-to-end with MONAI, containerized and served via FastAPI, and tracked in Weights & Biases."
Memorize that. It hits ~12 keywords from typical Bayern medtech postings.
Sources
[Roche Diagnostics — Alzheimer's blood biomarkers (NeuroToolKit, Penzberg Innovation Center Feb 2026)](https://www.roche.com/stories/alzheimers-blood-biomarkers)
[MONAI — open-source medical imaging framework + Model Zoo](https://monai.io/)
[VISTA3D unified 3D segmentation foundation model (arXiv)](https://arxiv.org/html/2406.05285v3)
[NVIDIA Tech Blog — Visual Foundation Models for Medical Image Analysis](https://developer.nvidia.com/blog/visual-foundation-models-for-medical-image-analysis/)
[Survival analysis using deep learning with medical imaging (PMC review)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11074924/)
[Frontiers — MCI to Alzheimer's via multimodal MRI and AI (2025 review)](https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2025.1596632/full)
[MIDL job board — DL in medical imaging postings](https://www.midl.io/job-board)
[deepc — Munich radiology AI platform careers](https://www.deepc.ai/about/career)
[Siemens Healthineers careers (Erlangen / global)](https://careers.siemens-healthineers.com/global/en)
[Munich ML jobs aggregate (Glassdoor, April 2026)](https://www.glassdoor.co.uk/Job/munich-machine-learning-jobs-SRCH_IL.0,6_IC4990924_KO7,23.htm)
[Eugene Yan — How to Interview ML/AI Engineers (hiring-manager perspective)](https://eugeneyan.com/writing/how-to-interview/)
[Imaging Wire — Top 2026 Radiology Trends](https://theimagingwire.com/2026/01/07/the-top-trends-shaping-radiology-in-2026/)
[AI in Medical Imaging Market Report 2026–2030 (Research and Markets)](https://www.businesswire.com/news/home/20260126040596/en/AI-in-Medical-Imaging-Market-Research-Report-2026-2030-Rising-Diagnostic-Volumes-Radiologist-Shortages-and-Telemedicine-Integration-Accelerate-Adoption---ResearchAndMarkets.com)