# F1 Race Predictor 🏎️

An end-to-end Machine Learning web application that predicts Formula 1 race finishes using historical data, qualifying performances, weather reports, circuit characteristics, and sprint results.

This project uses **FastF1** to ingest telemetry/session data, scikit-learn (RandomForest, GradientBoosting) to model probabilities, a **FastAPI** backend to run inference, and a **Vite/React** dark-themed dashboard to visualize predictions.

---

## 🌟 Key Features

*   **Robust Feature Engineering:** Incorporates 170+ features including weather conditions (rain, track temp), circuit technical metrics (corner counts, straights), sprint performances, and penalty-adjusted official grid starts.
*   **Offline Walk-Forward Training:** Memory-efficient architecture. Train models locally on your PC (where RAM is free), zip the models, and upload them to the web server.
*   **Ensemble ML Prediction:** Uses blended probabilities from Grid Position, Points Finish, DNF, and Multi-Class Outcome estimators to produce a unified prediction score.
*   **Data Ingestion API:** Hot-pull the latest race session data directly from the Ergrast API and FastF1.
*   **Modern Interactive Dashboard:** A responsive, dark-themed React application with probability bars, podium projections, and driver insights.
*   **AWS EKS Production-Ready:** Complete Docker multi-stage builds and Kubernetes infrastructure code included for high-availability deployment.

---

## 📂 Project Structure

```text
.
├── app/                  # FastAPI backend server
│   └── main.py           # API endpoints (/predict_race, /upload_models)
├── web/                  # React dashboard frontend
│   ├── src/              # React components & styles
│   ├── Dockerfile        # Nginx/Node deployment image
│   └── nginx.conf        # Nginx SPA config
├── k8s/                  # Kubernetes configuration manifests
├── src/                  # Core Python modules
│   └── inference/        # WalkForwardPredictor and active Inference Service
├── data/                 # Raw data (git-ignored)
├── models/               # .joblib ML models (git-ignored)
├── fastf1_csv_ingest.py  # Data extraction engine (pulls from F1 API)
├── retrain_offline.py    # Local script to train models and zip them for upload
├── train_*.py            # Standalone ML core model training algorithms
├── deploy.sh             # AWS Kubernetes automation script
├── docker-compose.yml    # Local containerized testing
└── requirements.txt      # Python dependencies
```

---

## 🛠️ Local Development Setup

### Option A: Fully Containerized (Docker Compose)
The easiest way to run the stack.
```bash
docker compose up --build
```
*   **Frontend:** `http://localhost:3000`
*   **Backend API:** `http://localhost:8080/docs`

### Option B: Native Execution

**1. Setup Python Backend:**
```bash
conda create -n f1_predict python=3.11 -y
conda activate f1_predict
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

**2. Setup React Frontend:**
```bash
cd web
npm install
npm run dev
# The site stays accessible on http://localhost:5173
```

---

## 🧠 Model Training & Upload Workflow

To save server memory (RAM), the API no longer trains models on the fly. Instead, use the offline retraining pipeline:

**1. Create a Walk-Forward model package:**
Run the offline training script locally. Specify the season and round you want the models to predict (they will be trained strictly on data *prior* to that race to stop target leakage).
```bash
python retrain_offline.py --season 2025 --round 5 --out models_upload.zip
```
*This step reads the dataset, trains 5 estimators, and saves them to `models_upload.zip`.*

**2. Hot-swap the web models:**
1. Open the React Dashboard.
2. Click **Upload Models**.
3. Select `models_upload.zip`. The server will dynamically extract the `.joblib` estimators and re-warm the memory cache without dropping API connections.

---

## 📥 Ingesting Data

If you need to pull the absolute latest F1 calendar and race timing data:
1. Open the dashboard.
2. Select your current Season.
3. Click the **📥 Ingest [Year] Data** button.

*Note: For the massive initial extraction of 2018-2025 data, use the python script directly:*
```bash
python fastf1_csv_ingest.py --overwrite
```

---

## 🌩️ Deployment (AWS EKS & ALB)

A complete highly-available Kubernetes architecture is bundled. The repository utilizes an AWS ALB for routing, ECR for container registries, and EKS for execution.

**Quick Deploy:**
Customize your Account ID and AWS Region variables inside `deploy.sh`.
```bash
# Provisions EKS Cluster, ECR Registries, and ALB Controller (takes ~20 mins)
./deploy.sh setup

# Builds the multi-stage Dockerfiles and pushes to Amazon ECR
./deploy.sh build

# Applies the K8s manifests (Deployments, HPA, Services, PVC)
./deploy.sh deploy
```

> ⚠️ Cost Warning: The deployed production EKS stack costs approximately ~$217/month. For cheaper hosting simply run `docker-compose up` on a $5/month VPS or a generic t3.small EC2.

---

## 🔄 CI/CD Automation

When code is committed to the `main` branch, the `.github/workflows/deploy.yml` action will trigger. It automatically builds the latest Docker images and dynamically patches the Kubernetes clusters.

*To enable actions: Provide `AWS_ROLE_ARN` as a repository secret inside GitHub.*
