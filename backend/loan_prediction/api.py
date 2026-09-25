from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .enhanced_service import enhanced_model_info, enhanced_ready, predict_enhanced
from .business_analysis import case_detail, optimize_policy, build_analysis_frame, run_business_analysis
from .schemas import (
    ApplicantInput,
    BatchPredictionRequest,
    EnhancedApplicantInput,
    EnhancedBatchPredictionRequest,
    EnhancedPredictionResponse,
    PredictionResponse,
    BusinessAnalysisScenario,
    PolicyOptimizationRequest,
)
from .service import model_info, predict

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Loan Approval Project API",
    version=__version__,
    description=(
        "Loan Approval Project turns a college Data Science internship prototype into a reproducible "
        "ML project. v1 preserves the original 614-row project; v2 is the expanded "
        "4,269-row high-capacity CIBIL/assets pipeline. Educational use only."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "enhanced_model_ready": enhanced_ready()}


# --------------------------- Business Analysis business API ---------------------------
@app.get("/api/analysis/status")
def analysis_status():
    frame = build_analysis_frame()
    return {
        "ready": True,
        "applications": int(len(frame)),
        "simulation": True,
        "description": "Business analytics and lending-operations simulation built on the project dataset.",
    }


@app.post("/api/analysis/run_business_analysis")
def analysis_run_business_analysis(payload: BusinessAnalysisScenario):
    try:
        return run_business_analysis(
            approve_threshold=payload.approve_threshold,
            reject_threshold=payload.reject_threshold,
            analysts=payload.analysts,
            productive_hours=payload.productive_hours,
            volume_multiplier=payload.volume_multiplier,
            segment_dimension=payload.segment_dimension,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/analysis/optimize-policy")
def analysis_optimize_policy(payload: PolicyOptimizationRequest):
    try:
        return optimize_policy(
            build_analysis_frame(),
            minimum_auto_agreement=payload.minimum_auto_agreement,
            analysts=payload.analysts,
            productive_hours=payload.productive_hours,
            volume_multiplier=payload.volume_multiplier,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/analysis/case/{application_id}")
def analysis_case(application_id: str):
    try:
        return case_detail(application_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Application not found") from exc


# ------------------------------ v2 enhanced API ------------------------------
@app.get("/api/v2/status")
def v2_status():
    return enhanced_model_info()


@app.get("/api/v2/model")
def get_v2_model_info():
    info = enhanced_model_info()
    if not info.get("ready"):
        raise HTTPException(status_code=503, detail=info)
    return info


@app.post("/api/v2/predict", response_model=EnhancedPredictionResponse)
def predict_v2(applicant: EnhancedApplicantInput):
    if not enhanced_ready():
        raise HTTPException(
            status_code=503,
            detail="Enhanced Loan Approval Model is not trained yet. Run: python backend/scripts/bootstrap_enhanced.py --quality max",
        )
    try:
        return predict_enhanced(applicant)
    except Exception as exc:
        logger.exception("Enhanced prediction request failed")
        raise HTTPException(status_code=500, detail="Enhanced prediction failed") from exc


@app.post("/api/v2/predict/batch")
def predict_v2_batch(payload: EnhancedBatchPredictionRequest):
    if not enhanced_ready():
        raise HTTPException(status_code=503, detail="Enhanced Loan Approval Model model is not trained")
    try:
        return {"predictions": [predict_enhanced(applicant) for applicant in payload.applicants]}
    except Exception as exc:
        logger.exception("Enhanced batch prediction failed")
        raise HTTPException(status_code=500, detail="Enhanced batch prediction failed") from exc


# ------------------------------ v1 legacy API -------------------------------
@app.get("/api/v1/model")
def get_model_info():
    try:
        return model_info()
    except Exception as exc:
        logger.exception("Legacy model-info request failed")
        raise HTTPException(status_code=503, detail="Legacy model information unavailable") from exc


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_one(applicant: ApplicantInput):
    try:
        return predict(applicant)
    except Exception as exc:
        logger.exception("Legacy prediction request failed")
        raise HTTPException(status_code=500, detail="Legacy prediction failed") from exc


@app.post("/api/v1/predict/batch")
def predict_batch(payload: BatchPredictionRequest):
    try:
        return {"predictions": [predict(applicant) for applicant in payload.applicants]}
    except Exception as exc:
        logger.exception("Legacy batch prediction request failed")
        raise HTTPException(status_code=500, detail="Legacy batch prediction failed") from exc


FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    assets = FRONTEND_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        root = FRONTEND_DIST.resolve()
        candidate = (FRONTEND_DIST / full_path).resolve()
        if full_path and candidate.is_relative_to(root) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(root / "index.html")
