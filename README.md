# GameSense FC — Dual-Framework Recommender (PyTorch + TensorFlow)

A small, production-style project that simulates **EA SPORTS FC** personalization:
- **PyTorch NCF** baseline for implicit feedback
- **TensorFlow Wide & Deep** baseline
- Synthetic user sessions + player “content” with room to plug real FIFA data
- Simple **Streamlit** demo

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) make synthetic data
python scripts/generate_synthetic.py --n_users 1500 --n_items 1200 --seed 42

# 2) train baselines
python train_pytorch.py
python train_tensorflow.py

# 3) run demo
streamlit run demo/app.py
