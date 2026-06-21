# Presentation Plan — Slide_ASL_EN.pptx

Target length: ~20–22 min talk + Q&A (23 slides). Timings are guidelines — the
3 starred slides (★) carry the actual argument; protect their time first if
you're running short. Slides marked *(skip if short on time)* can be cut
without breaking the story.

The whole talk answers ONE question, repeated at the start and the end:
**"Does a smarter representation actually help, or does it just add a new
way to fail?"** Every section should visibly pay off that question.

---

## 1. Title (0:30)
Just state the project and team. Don't read the subtitle aloud.

## 2. Motivation (1:00)
- Open with the *why*: ASL is a real communication need, not a toy task.
- Point at the montage: "these are real test images — note the lighting/background varies."
- Land the core question as a quote, slowly: *"does a smarter representation actually improve recognition, or just add a new point of failure?"* — this is the thesis the whole deck proves.

## 3. Objective & Contributions (1:00)
- Restate the research question in one breath, then move fast through the 3 boxes.
- Say "modular" out loud and gesture at the word — it's the reason 13 pipelines could be compared fairly at all.

## 4. Dataset Overview (1:00)
- Table: just say "train set to fit models, test_split to evaluate everything — 2,600 images, 26 classes."
- Point at the UMAP/t-SNE plot: "even something as simple as 42 hand-landmark numbers already separates all 26 letters cleanly — that's a preview, we'll come back to this plot."
- One sentence on why this matters: clean studio photos → expect very high accuracy → that's exactly why clean accuracy alone won't be enough to tell pipelines apart later.

## 5. Methodology: Two-Module Architecture (1:00)
- Read the equation left to right once: raw image → Representation → Recognizer → label.
- Say explicitly: "we kept these two modules independent on purpose, so we could swap either one and measure the effect in isolation."

## 6. Module 1 — Representation ★ light pass (1:15)
- Don't narrate every cell. Pick ONE row from the grid and walk it across columns: "this hand, raw, then cropped, then CLAHE'd, then sharpened."
- Point at a black MediaPipe cell: "and here — MediaPipe simply lost the hand. Keep this in mind, it comes back."
- This plants the seed for the case study later — don't over-explain yet.

## 7. Module 2 — Recognizer (1:00)
- One line per row max: "two image-based recognizers — a small fine-tuned CNN and a large pretrained ViT — and three landmark-based ones using the same 42 numbers."
- Flag the expectation, not yet the result: "pretrained ViT should, in theory, win on both accuracy and robustness."

## 8. Evaluation Metrics ★ (1:30)
- This is the slide that makes the case study land later — don't rush it.
- Walk the two formulas literally: "the codebase's accuracy only counts images where a hand was found. If MediaPipe misses 17% of hands, those 17% are quietly dropped, not counted as wrong."
- Say the punchline once, calmly: "that means a pipeline can report a beautiful accuracy number while still failing on real input."
- Briefly name the 8 corruption types as a list, don't dwell.

## 9. Quantitative Results table (1:30)
- Don't read the whole table. Guide the eye top to bottom: "best clean accuracy at the top, all non-MediaPipe."
- Point at the two bolded rows + the red 0.0000 column: "every MediaPipe pipeline — bold rows — collapses to zero on the worst corruption. Including the one with 98.5% clean accuracy."
- This is the table version of the thesis. Say: "keep this row in mind" (mediapipe_crop_vit) — it returns in the case study.

## 10–12. Three comparison charts (1:00 each = 3:00)
- Accuracy chart: "same story as the table, visually — MediaPipe pipelines visibly lower / more scattered."
- Latency chart: "but MediaPipe and landmark pipelines aren't worthless — they're fast. This is a real trade-off, not a pure loss."
- Robustness chart: "this is the chart that matters most — watch how far some bars drop under corruption."

## 13–14. Qualitative grids (1:00 each = 2:00)
- Representation grid: "5 real images, every representation type, side by side. Black cells = hand not found."
- Recognizer grid: "now the full pipelines — green is correct, red is wrong. Notice the red cells cluster in the exact same rows across every MediaPipe-based column — one upstream failure, repeated downstream."

## 15. Case Study ★★★ THE KEY SLIDE (2:00)
- Slow down here. This is the payoff of slides 6, 8, 9, and 14 — say so explicitly: "remember the metric problem from slide 8? Here's what it looks like on one real image."
- Walk the arithmetic out loud: "98.5% reported. 2110 out of 2600 actually correct. That's 81%. Seventeen points gone — not because the model is bad, because the metric hid a detector failure."
- End with the contrast: "and the simplest pipeline in our study — raw image, no detection step — got this exact image right, 100% confidence."
- Pause after this slide. Let it sit before moving on.

## 16. Grad-CAM (1:00)
- Quick, visual: "this is what the network is 'looking at' — the heat concentrates on the hand both times, so the classifier itself is sound; the failures are upstream, in detection, not in the CNN."

## 17. Feature Space — CNN Embeddings by Class (0:45)
- "Same story as the landmark plot from the Dataset slide, now in the CNN's own 512-dimensional space — classes separate just as cleanly, so when the pipeline *does* see the hand, classification itself is not the bottleneck."

## 18. More Qualitative Examples *(skip if short on time)* (0:30)
- One sentence: "to be fair — most images are easy. Every pipeline gets these right. The failures are concentrated in specific lighting/angle conditions, not everywhere."

## 19. Feature Space — Representation Shift (1:00)
- "Same images, four representations, one shared backbone. Enhancement (CLAHE/Gamma) barely moves the point in feature space — but cropping does, visibly, to a different region."
- Tie back: "this is the quantitative reason a crop-based representation behaves like a different problem to the recognizer, not just a cleaner version of the same one."

## 20. Conclusion ★ (1:30)
- Three boxes, one breath each. End on the third: "simpler can be more robust — that directly answers the question from slide 2."
- State limitations honestly in one sentence; don't apologize at length.
- Future work: one sentence, e.g. "the most actionable fix is a raw-image fallback whenever MediaPipe fails."

## 21. Work Assignment (0:30)
Read names + one phrase per person, no elaboration.

## 22. References *(skip narrating, leave on screen 5s if asked)*

## 23. Thank You (0:30)
- Repeat the one-sentence takeaway from the slide text — this is what the audience should remember if they remember nothing else.
- Invite questions.

---

## Anticipated Questions & Suggested Answers

**Q: Why not just always use the ViT/SigLIP backbone if it's better?**
A: It is more accurate and more robust, but ~15–25× slower (compare FPS column) — for real-time webcam use, ResNet18-on-raw or landmark-based pipelines are the practical choice; ViT is the right call when latency doesn't matter.

**Q: Why does MediaPipe fail on 17% of images?**
A: Mostly low-contrast hand-vs-background and unusual lighting in the test set — MediaPipe's detector confidence threshold rejects ambiguous detections rather than guessing.

**Q: Could you fix the MediaPipe failure rate?**
A: Yes — e.g. lowering the detection confidence threshold (trading false negatives for false positives), or adding a raw-image fallback recognizer when no hand is detected, instead of returning "Unknown."

**Q: Is this evaluated on only one dataset?**
A: Yes — ASL Alphabet (Kaggle), studio-style. This is a named limitation; real-world webcam footage would likely show larger gaps between pipelines.

**Q: What's the single most important takeaway?**
A: Always report end-to-end accuracy that counts detection failures as wrong, and validate robustness — not just clean accuracy — before claiming a pipeline is "better."
