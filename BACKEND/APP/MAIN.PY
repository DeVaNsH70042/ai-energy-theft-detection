from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.database import engine, Base, SessionLocal
import app.models as models
from app.models import RawReadingCreate
from app.ml_engine import train_model, score_reading


app = FastAPI(title="AI Energy Theft Detection API")

# Create tables
Base.metadata.create_all(bind=engine)

# Train ML model once at startup
train_model()


# ------------------ DB Dependency ------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------ Health Endpoint ------------------
@app.get("/health")
def health_check():
    return {"status": "running"}


# ------------------ Main Reading Endpoint ------------------
@app.post("/readings")
def create_reading(reading: RawReadingCreate, db: Session = Depends(get_db)):

    # -------- PHYSICS LAYER --------
    if reading.actual_power != 0:
        mismatch_ratio = (
            (reading.actual_power - reading.meter_power)
            / reading.actual_power
        )
    else:
        mismatch_ratio = 0

    if mismatch_ratio < 0.05:
        risk_score = 10
        single_status = "Normal"
    elif mismatch_ratio < 0.15:
        risk_score = 50
        single_status = "Suspicious"
    else:
        risk_score = 90
        single_status = "High Risk"

    # -------- STORE IN DATABASE --------
    db_reading = models.RawReading(
        meter_id=reading.meter_id,
        actual_power=reading.actual_power,
        meter_power=reading.meter_power,
        mismatch_ratio=mismatch_ratio,
        timestamp=reading.timestamp
    )

    db.add(db_reading)
    db.commit()
    db.refresh(db_reading)

    # -------- ROLLING WINDOW LOGIC --------
    last_readings = (
        db.query(models.RawReading)
        .filter(models.RawReading.meter_id == reading.meter_id)
        .order_by(models.RawReading.timestamp.desc())
        .limit(5)
        .all()
    )

    mismatch_history = [r.mismatch_ratio for r in last_readings]

    high_risk_count = sum(
        1 for value in mismatch_history if value > 0.15
    )

    # -------- ML LAYER --------
    ml_flag = score_reading(mismatch_ratio)

    # -------- HYBRID DECISION ENGINE --------
    if high_risk_count >= 3 or ml_flag == 1:
        final_status = "CONFIRMED THEFT"
    else:
        final_status = "Monitoring"

    return {
        "message": "Reading stored",
        "mismatch_ratio": mismatch_ratio,
        "single_status": single_status,
        "risk_score": risk_score,
        "high_risk_in_last_5": high_risk_count,
        "ml_flag": ml_flag,
        "final_status": final_status
    }