# GameSense FC — Dual-Framework EA SPORTS FC Recommender

A dual-framework (PyTorch & TensorFlow) recommender system inspired by **EA SPORTS FC** personalization workflows.  
The project simulates **personalized content recommendations** for players using **synthetic gameplay data** and real FIFA player stats (optional).  

It’s designed to:
- Showcase **deep learning personalization** in **both PyTorch and TensorFlow**
- Demonstrate **A/B testing and evaluation metrics**
- Provide an **interactive Streamlit demo** for recruiters and hiring managers

---

## Project Plan

### **Phase 1 — Data Foundation** 
- Generate synthetic user sessions (`scripts/generate_synthetic.py`)
- Include user playstyle, preferred positions, and item (player) attributes
- Save train/val/test splits in `data/processed/`

### **Phase 2 — Baseline Models** 
- **PyTorch NCF** (Neural Collaborative Filtering) — `train_pytorch.py`
- **TensorFlow Wide & Deep** — `train_tensorflow.py`
- Shared evaluation: Precision@K, Recall@K, NDCG@K

### **Phase 3 — Demo UI** 
- Streamlit app in `demo/app.py`
- Allows selecting a user and viewing Top-K recommendations

### **Phase 4 — Enhancements** (Planned)
- Integrate real FIFA player data from Kaggle
- Add contextual features (match type, formation, skill rating)
- Implement **sequence models** (e.g., SASRec, GRU4Rec)
- Build A/B testing simulator for model comparison
- Deploy demo online (Streamlit Cloud / Hugging Face Spaces)

---

## Tech Stack
- **Python**
- **PyTorch** & **TensorFlow/Keras**
- **Pandas**, **NumPy**, **scikit-learn**
- **Streamlit**
- **AWS SageMaker** (planned integration)

---

## Quickstart

**Clone & Install**
```bash
git clone https://github.com/<your-username>/ea-fc-recommender-dualframework.git
cd ea-fc-recommender-dualframework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
