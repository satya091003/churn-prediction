## 📊 Model Performance

Trained and evaluated on a stratified 80/20 split. **PR-AUC** (not accuracy) is the primary metric — churn is imbalanced (~21% positive class), so a model that predicts "no churn" for everyone would score 80%+ accuracy while being useless.

| Model | ROC-AUC | PR-AUC | F1 (churn class) |
|---|---|---|---|
| **Logistic Regression** ⭐ | 0.780 | **0.488** | 0.521 |
| Random Forest | 0.768 | 0.464 | 0.510 |
| Gradient Boosting | 0.762 | 0.446 | 0.352 |

⭐ = selected as the best model by PR-AUC

*(Swap in the real Kaggle Telco dataset for stronger, more representative numbers — see [Dataset](#-dataset) below.)*

---

## 💰 Business Impact (sample output)

Running the model on the sample dataset (7,000 customers) surfaces:

| Segment | Customers | Est. Monthly Revenue at Risk | Recommended Action |
|---|---|---|---|
| 🔴 Critical Risk | 1,286 | $92,425 | Immediate outreach + retention offer within 48h |
| 🟠 High Risk | 2,305 | $141,435 | Proactive check-in, loyalty discount |
| 🟡 Watch | 1,685 | $94,748 | Add to nurture campaign, monitor next cycle |
| 🟢 Healthy | 1,724 | $83,873 | No action — upsell / referral candidate |

This is the difference between "the model says 21% will churn" and "here are 1,286 specific customers, why they're at risk, and what to do about each one this week."
