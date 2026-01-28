# main.py
import traceback

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from scoring import score_monthly_simple

app = FastAPI(debug=True) 


class ScoreRequest(BaseModel):
    project_id: str
    client_key: str
    target_table: str
    months_back: int = 24
    aggregate_last_n: int = 12
    rrss_policy: str = "renormalize"
    ga_fallback_aov: float = 0.0
    rrss_blend_policy: str = "split"


@app.post("/score")
def score(req: ScoreRequest):
    try:
        value = score_monthly_simple(
            project_id=req.project_id,
            client_key=req.client_key,
            target_table=req.target_table,
            months_back=req.months_back,
            aggregate_last_n=req.aggregate_last_n,
            rrss_policy=req.rrss_policy,
            ga_fallback_aov=req.ga_fallback_aov,
            rrss_blend_policy=req.rrss_blend_policy,
        )
        return {"status": "ok", "score": value, "client_key": req.client_key}
    except Exception as e:
        traceback.print_exc()  
        raise HTTPException(status_code=500, detail=str(e))
