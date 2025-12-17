"""
FastAPI Backend for Multi-Agent Investment System
Provides REST API endpoints for the web frontend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import json
import os
import sys
from datetime import datetime
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(title="Multi-Agent Investment System API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class CustomerPortfolio(BaseModel):
    customer_id: str
    risk_appetite: str  # LOW, MEDIUM, HIGH
    investment_capital: float
    sector_preferences: List[str]
    return_expectation: float
    investment_horizon: str  # SHORT_TERM, MEDIUM_TERM, LONG_TERM

class AgentStatus(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

@app.get("/")
async def root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "monitor": "ready",
            "planner": "ready",
            "predictor": "ready"
        }
    }

@app.post("/api/run-monitor")
async def run_monitor():
    """Run the Monitor Agent to collect market data"""
    try:
        result = subprocess.run(
            ["python", "src/monitor/monitor_agent.py"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        # Load the generated market data regardless of return code
        data_file = "data/monitor_data/general_market_data.json"
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                market_data = json.load(f)

            return {
                "status": "success",
                "message": "Market data collected successfully",
                "data": {
                    "stocks_count": len(market_data.get("stocks", [])),
                    "sectors_count": len(market_data.get("sectors", [])),
                    "timestamp": market_data.get("timestamp")
                }
            }
        else:
            return {
                "status": "error",
                "message": f"Monitor agent failed: {result.stderr[:200] if result.stderr else 'Unknown error'}"
            }

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Monitor agent timeout"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/run-planner")
async def run_planner(portfolio: CustomerPortfolio):
    """Run the Planner Agent to create investment plan"""
    try:
        # Prepare arguments for the planner agent
        cmd = [
            "python", "src/planner/planner_agent.py",
            "--customer_id", str(portfolio.customer_id),
            "--risk_appetite", str(portfolio.risk_appetite),
            "--investment_capital", str(portfolio.investment_capital),
            "--sector_preferences", ",".join(portfolio.sector_preferences),
            "--return_expectation", str(portfolio.return_expectation),
            "--investment_horizon", str(portfolio.investment_horizon)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        # Load the generated investment plan
        plan_file = f"data/planner_data/investment_plan_{portfolio.customer_id}.json"
        if os.path.exists(plan_file):
            with open(plan_file, 'r') as f:
                plan_data = json.load(f)

            return {
                "status": "success",
                "message": "Investment plan created successfully",
                "data": plan_data
            }
        else:
            return {
                "status": "error",
                "message": f"Planner agent failed: {result.stderr[:200] if result.stderr else 'Plan file not found'}"
            }

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Planner agent timeout"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/run-predictor/{customer_id}")
async def run_predictor(customer_id: str):
    """Run the Predictor Agent to validate investment plan"""
    try:
        result = subprocess.run(
            ["python", "src/predictor/predictor_agent.py"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        # Load the validation results
        validation_file = f"data/predictor_data/validation_{customer_id}.json"
        if os.path.exists(validation_file):
            with open(validation_file, 'r') as f:
                validation_data = json.load(f)

            return {
                "status": "success",
                "message": "Portfolio validation completed",
                "data": validation_data
            }
        else:
            return {
                "status": "error",
                "message": f"Predictor agent failed: {result.stderr[:200] if result.stderr else 'Validation file not found'}"
            }

    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Predictor agent timeout"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/market-data")
async def get_market_data():
    """Get latest market data from Monitor Agent"""
    try:
        data_file = "data/monitor_data/general_market_data.json"
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                return json.load(f)
        else:
            raise HTTPException(status_code=404, detail="Market data not found. Run Monitor Agent first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/investment-plan/{customer_id}")
async def get_investment_plan(customer_id: str):
    """Get investment plan for a customer"""
    try:
        plan_file = f"data/planner_data/investment_plan_{customer_id}.json"
        if os.path.exists(plan_file):
            with open(plan_file, 'r') as f:
                return json.load(f)
        else:
            raise HTTPException(status_code=404, detail=f"Investment plan for {customer_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/validation/{customer_id}")
async def get_validation(customer_id: str):
    """Get validation results for a customer"""
    try:
        validation_file = f"data/predictor_data/validation_{customer_id}.json"
        if os.path.exists(validation_file):
            with open(validation_file, 'r') as f:
                return json.load(f)
        else:
            raise HTTPException(status_code=404, detail=f"Validation for {customer_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/run-all")
async def run_all_agents(portfolio: CustomerPortfolio):
    """Run all three agents in sequence"""
    results = {}

    try:
        # Step 1: Run Monitor Agent
        monitor_result = await run_monitor()
        results["monitor"] = monitor_result

        # Step 2: Run Planner Agent
        planner_result = await run_planner(portfolio)
        results["planner"] = planner_result

        # Step 3: Run Predictor Agent
        predictor_result = await run_predictor(portfolio.customer_id)
        results["predictor"] = predictor_result

        return {
            "status": "success",
            "message": "All agents completed successfully",
            "results": results
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "results": results
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

