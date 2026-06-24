# 🚦 LionCity Traffic Intelligence

A real-time traffic congestion prediction platform for Singapore, built with a Random Forest classifier and explainable AI (SHAP), wrapped in an interactive Streamlit interface.

🔗 **Live App:** [lioncity-traffic-intelligence.streamlit.app](https://lioncity-traffic-intelligence.streamlit.app/)
🔗 **Research & Notebooks:** https://github.com/Abhishah2004/LionCity-Traffic-Intelligence-Jupyter-Notebooks

---

## 📌 Overview

LionCity Traffic Intelligence predicts traffic congestion levels (**Low / Medium / High**) across Singapore based on time, weather, tourism, and event-related factors — without using vehicle speed as an input, since speed is a *consequence* of congestion rather than a predictive cause. Instead, expected speed ranges are derived from historical data and shown alongside each prediction.

The app simulates "what-if" traffic scenarios and explains *why* the model made a given prediction using SHAP, while also visualizing affected zones (school areas, festivals, and events) on an interactive Singapore map.

---

## ✨ Features

- **Scenario simulation** — adjust hour, day, month, weather, tourism index, and special events to see real-time congestion predictions
- **Confidence & probability breakdown** — full Low/Medium/High probability distribution for every prediction, not just the top class
- **Expected speed range** — percentile-based (25th–75th) speed estimate per predicted congestion level
- **SHAP explainability** — a live, per-prediction chart showing which features pushed the result toward or away from the predicted class
- **Interactive map** — Singapore choropleth with school-zone markers (auto-derived for 7–9 AM on school days) and a 60+ event-to-location mapping for festivals, concerts, and public holidays
- **Key conditions panel** — flags active conditions (rush hour, weekend, holiday, heavy rainfall, school hours, events) at a glance

---

## 🧠 Model

- **Algorithm:** Random Forest Classifier (`n_estimators=300`, `max_depth=10`)
- **Accuracy:** 85.5%
- **Features (13):** hour, month, day of week, weekend/holiday flags, school-hour flag (auto-derived), workday flag, temperature, rainfall, humidity, wind speed, air quality index, tourist arrival index, special event type
- **Explainability:** SHAP `TreeExplainer`, computed live per prediction

---

## 🛠️ Tech Stack

`Python` · `Streamlit` · `Scikit-learn` · `SHAP` · `Plotly` · `Pandas` · `NumPy`

---

## 📂 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── rf_model.pkl             # Trained Random Forest classifier
├── le.pkl                   # Label encoder for congestion levels
├── oe.pkl                   # Ordinal encoder for categorical features
├── speed_ranges.pkl         # Precomputed speed ranges per congestion level
├── day_options.pkl          # Valid day-of-week values
├── event_options.pkl        # Valid special event types
├── shap_background.pkl      # Background sample for SHAP TreeExplainer
├── singapore.geojson         # Singapore planning area boundaries (map overlay)
├── merlion.jpg               # Hero banner image
└── requirements.txt
```




---

## 📝 Notes & Design Decisions

- **Vehicle speed is intentionally excluded** as a model input to avoid circular reasoning — a user predicting congestion wouldn't already know their speed. Speed ranges are instead derived post-hoc from historical data per congestion class.
- **School-hour flag is auto-derived** (active 7–9 AM, skipped on weekends/public holidays) rather than user-toggled, for more realistic scenario simulation.
- **Event-to-location mapping is a static, manually curated lookup** based on publicly known Singapore event venues — not a live geocoding system. All zones shown for a selected event are colored by the single scenario-level prediction, since the model predicts congestion for the overall scenario rather than per-location.

---

## 📄 License

MIT
