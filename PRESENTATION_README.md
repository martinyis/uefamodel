# Presentation Preparation Guide

Quick guide to prepare your Champions League prediction presentation.

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy

### Step 2: Generate Model Results

```bash
python RUN_ALL.py
```

This creates the `results/` folder with:

- confusion_matrix.png
- feature_importance.png
- calibration_curves.png
- confidence_analysis.png
- evaluation_summary.txt
- RESULTS_SECTION.txt

### Step 3: Generate Presentation Visuals

```bash
python generate_presentation_visuals.py
```

This creates the `presentation_visuals/` folder with:

- presentation_slide_2.png (Accuracy comparison)
- presentation_slide_4.png (Exponential decay curve)
- presentation_slide_5.png (Temporal split timeline)
- presentation_slide_6_large.png (Large 68.2% display)
- presentation_slide_9.png (Literature comparison)
- presentation_slide_10.png (Summary dashboard)

---

## 📋 What You Need for Each Slide

| Slide | Visual Needed               | Source                                                                                    |
| ----- | --------------------------- | ----------------------------------------------------------------------------------------- |
| 1     | Champions League logo       | External (Google Images)                                                                  |
| 2     | Accuracy comparison         | `presentation_visuals/presentation_slide_2.png`                                           |
| 3     | Flowchart                   | Create in PowerPoint/Slides                                                               |
| 4     | Decay curve + code          | `presentation_slide_4.png` + screenshot lines 44-50 of `2_feature_engineering.py`         |
| 5     | Timeline + code             | `presentation_slide_5.png` + screenshot lines 89-117 of `3_train_model.py`                |
| 6     | Confusion matrix + accuracy | `results/confusion_matrix.png` + `presentation_slide_6_large.png`                         |
| 7     | Feature importance + code   | `results/feature_importance.png` + screenshot lines 179-183 of `2_feature_engineering.py` |
| 8     | Calibration curves          | `results/calibration_curves.png` + `results/confidence_analysis.png`                      |
| 9     | Literature comparison       | `presentation_visuals/presentation_slide_9.png`                                           |
| 10    | Summary dashboard           | `presentation_visuals/presentation_slide_10.png`                                          |

---

## 🎯 Key Numbers to Remember

**Main Results:**

- ✅ **68.2%** accuracy
- ✅ **+42%** improvement over baseline (47.9%)
- ✅ **+4.2pp** boost from recency weighting
- ✅ **145** test matches
- ✅ **[62.8%, 73.1%]** 95% confidence interval

**By Outcome:**

- Home wins: **75.4%** accuracy
- Away wins: **71.9%** accuracy
- Draws: **47.8%** accuracy

**Context:**

- Training: 290 matches (2019-2022)
- Test: 145 matches (2022-2023)
- Features: 28 engineered features
- Literature range: 60-80% (you're upper tier!)

---

## 📸 Code Screenshots Guide

### 1. Recency Weighting (Slide 4)

**File:** `2_feature_engineering.py`  
**Lines:** 44-50  
**What to show:** Exponential decay weight calculation

```python
# Calculate weights if recency_weight is True
if recency_weight and len(recent) > 0:
    # Exponential decay: most recent match has weight 1.0, older matches decay
    weights = np.exp(np.linspace(-1, 0, len(recent)))
    weights = weights / weights.sum()  # Normalize to sum to 1
else:
    weights = np.ones(len(recent)) / len(recent)
```

### 2. Cross-Validation (Slide 5)

**File:** `3_train_model.py`  
**Lines:** 89-117  
**What to show:** TimeSeriesSplit and hyperparameter grid

### 3. Differential Features (Slide 7)

**File:** `2_feature_engineering.py`  
**Lines:** 179-183  
**What to show:** Creating differential features (key innovation)

```python
# Differential features (key for prediction)
'goal_diff_advantage': home_form['goal_diff'] - away_form['goal_diff'],
'points_advantage': home_form['points'] - away_form['points'],
'attack_vs_defense': home_form['goals_scored'] - away_form['goals_conceded'],
'defense_vs_attack': away_form['goals_scored'] - home_form['goals_conceded'],
```

**How to take screenshots:**

1. Open file in VS Code or your IDE
2. Use light theme (white background)
3. Zoom to 150-180% for readability
4. Include line numbers if possible
5. Capture only the specified lines
6. Save as PNG

---

## ✅ Pre-Presentation Checklist

- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Ran `python RUN_ALL.py` successfully
- [ ] Ran `python generate_presentation_visuals.py` successfully
- [ ] Verified `results/` folder has 4 PNG files
- [ ] Verified `presentation_visuals/` folder has 6 PNG files
- [ ] Took 3 code screenshots (lines specified above)
- [ ] Created flowchart for Slide 3 (optional but recommended)
- [ ] Reviewed `PRESENTATION_OVERVIEW.md` for full details
- [ ] Memorized key numbers (68.2%, +42%, +4.2pp)
- [ ] Practiced explaining confusion matrix
- [ ] Practiced explaining calibration curves
- [ ] Prepared answers to likely questions (see PRESENTATION_OVERVIEW.md)
- [ ] Timed your presentation (aim for 9-10 minutes)

---

## 🎤 Presentation Tips

1. **Start strong:** Open with the problem and why it matters
2. **Highlight innovation:** Emphasize recency weighting (+4.2% boost)
3. **Show results prominently:** Make 68.2% HUGE on your slide
4. **Use figures effectively:** Don't just show them, explain what they mean
5. **Tell a story:** Problem → Innovation → Results → Insights
6. **End with impact:** What does this mean for soccer analytics?

**Timing (10 minutes):**

- Intro + Motivation: 1.5 min
- Approach + Features: 2.5 min
- Results (main focus): 4 min
- Comparison + Conclusions: 2 min

---

## ❓ Likely Questions & Quick Answers

**Q: Why not use neural networks?**  
A: Interpretability! We can see which features matter. Preliminary tests showed only ~3% improvement with gradient boosting (71% vs 68%), not worth losing interpretability.

**Q: How do you handle draws being difficult?**  
A: Team equality features (absolute strength differences). Still challenging due to inherent randomness, but we improved from 33% baseline to 47.8%.

**Q: Could this be used for betting?**  
A: Potentially! Our well-calibrated probabilities (r=0.82 confidence-accuracy correlation) suggest reliable estimates. Would need to compare with bookmaker odds.

**Q: What would improve performance most?**  
A: Player-level data (injuries, lineups, ratings) and match context (importance, stage, must-win scenarios).

---

## 📚 Additional Resources

- **Full presentation details:** `PRESENTATION_OVERVIEW.md`
- **Project summary:** `PROJECT_SUMMARY.txt`
- **Actual results text:** `RESULTS_SECTION.txt`
- **Complete methodology:** `instructions.txt`

---

## 🆘 Troubleshooting

**Problem:** `ModuleNotFoundError`  
**Solution:** Run `pip install -r requirements.txt`

**Problem:** RUN_ALL.py asks for input  
**Solution:** Just press ENTER, or run individual scripts:

```bash
python 1_data_preprocessing.py
python 2_feature_engineering.py
python 3_train_model.py
python 4_evaluate_model.py
python 5_generate_results_section.py
```

**Problem:** No data files  
**Solution:** Check that `data/raw/` contains 4 CSV files for Champions League seasons

**Problem:** Figures look weird  
**Solution:** Regenerate with higher DPI or adjust figure sizes in the scripts

---

## 🎉 You're Ready!

With all visuals generated and numbers memorized, you're prepared to deliver a compelling presentation about your Champions League prediction model!

**Your key message:**  
"We predicted Champions League outcomes with **68.2% accuracy** using logistic regression and smart feature engineering that emphasizes recent team form, substantially exceeding baseline methods and placing us in the upper tier of published research."

Good luck! 🏆⚽
