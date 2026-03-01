from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from app.database import Base

class RawReading(Base):
    __tablename__ = "raw_readings"

    id = Column(Integer, primary_key=True, index=True)
    meter_id = Column(String, index=True)
    actual_power = Column(Float)
    meter_power = Column(Float)
    mismatch_ratio = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

from pydantic import BaseModel
from datetime import datetime

class RawReadingCreate(BaseModel):
    meter_id: str
    actual_power: float
    meter_power: float
    timestamp: datetime