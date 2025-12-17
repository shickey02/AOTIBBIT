# bbit_geomlang

A small prototype exploring geometric "language" recognition with a simple neural network.

Files
- `geomlang_prototype.py` — main prototype: synthetic data, MLP model, training loop, occlusion tests.
- `requirements.txt` — basic dependencies (`torch`, `numpy`, `matplotlib`).

Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Run prototype (small experiment):

```powershell
python geomlang_prototype.py --epochs 30
```

Notes & Future Ideas
- Replace synthetic circle task with more complex geometric primitives (triangles, polygons).
- Replace MLP with an equivariant or graph-based model to better capture geometry.
- Add interpretability methods (feature attribution, saliency) and richer occlusion tests.
- Save checkpoints and add CLI flags for experiment logging.

