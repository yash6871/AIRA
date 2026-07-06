# AIRA — Lightweight, Cloud + Mobile Ready

## Kya badla hai
- `insightface` hi face **detect + recognize** dono karta hai ab (yolo_detector.py, faiss_db.py, face_arcface.py hata diye)
- `torch`, `ultralytics`, `faiss-cpu` hata diye — inhi ki wajah se size aur Windows build error aa rahe the
- `cv2.VideoCapture` / `cv2.imshow` (sirf laptop pe chalne wala popup camera) hata diya
- Naya camera flow: **browser ka `getUserMedia`** (`static/js/camera.js`) — ye laptop webcam aur phone camera dono pe automatically kaam karta hai, alag code nahi likhna padta
- `/register` — ab camera se seedha browser me photo capture hoti hai
- `/live_attendance` — naya page, browser camera se har 1.5 sec me frame bhejta hai aur attendance mark karta hai
- Total project size: ~14MB code + insightface models runtime pe download hote hain (~330MB, sirf pehli baar)

## Laptop pe local run karna (Windows)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
Browser me kholo: http://localhost:5000
(Pehli baar chalane pe insightface apne face models download karega — internet chahiye hoga usi ek baar ke liye)

⚠️ Agar Windows pe `pip install insightface` fir se C++ build error de, to ye chalao (Linux/Render pe ye zarurat nahi padegi):
https://visualstudio.microsoft.com/visual-cpp-build-tools/ (sirf "Desktop development with C++" workload install karo)

## Render pe FREE deploy karna
1. Is poore folder ko GitHub repo bana ke push karo
2. https://render.com pe jao → New → Web Service → apna GitHub repo select karo
3. Render `render.yaml` khud detect kar lega (build/start command already set hain)
4. Deploy dabao — 5-10 min me live ho jayega, free HTTPS URL milega
5. Wahi URL phone ke browser me kholo — camera turant kaam karega (HTTPS hone ki wajah se)

Agar `render.yaml` detect na ho to manually set karo:
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`

## Vercel ke baare me note
Vercel ka free serverless function limit **250MB unzipped** hai — insightface + onnxruntime + numpy + opencv isse jyada ho jaate hain, isliye backend Vercel pe fit nahi hoga. **Render free tier hi sahi choice hai** is project ke liye.

## Accuracy vs Size trade-off
`face_engine.py` me env variable `FACE_MODEL` set kar sakte ho:
- `buffalo_l` (default) → best accuracy, ~330MB model (Render pe koi dikkat nahi)
- `buffalo_s` → chhota (~50MB), thodi kam accuracy — agar kabhi disk/RAM tight lage to switch karo

## Important limitation
- SQLite database (`aira.db`) Render free tier pe **ephemeral** hai — matlab agar service restart/redeploy ho, DB reset ho sakta hai. Production ke liye Render ka free **PostgreSQL** addon use karna better hoga (abhi ke liye maine nahi badla, taaki existing sqlite code chalta rahe — bolo to wo bhi kar dunga).
