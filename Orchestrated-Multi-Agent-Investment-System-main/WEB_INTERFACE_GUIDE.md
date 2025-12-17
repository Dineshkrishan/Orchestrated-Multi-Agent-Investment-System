# Multi-Agent Investment System - Web Interface Guide

## 🎉 System Overview

The Multi-Agent Investment System now has a **complete web interface** with a FastAPI backend and responsive HTML/CSS/JavaScript frontend!

## 📁 New Files Created

### Backend
- **`app.py`** - FastAPI server with REST API endpoints

### Frontend
- **`static/index.html`** - Main web interface with 4 tabs
- **`static/styles.css`** - Responsive styling with gradient design
- **`static/script.js`** - Frontend logic for API calls and UI updates

### Testing
- **`test_system.py`** - Automated test script for all API endpoints

## 🚀 How to Run

### 1. Start the Server

```bash
python app.py
```

The server will start on **http://localhost:8000**

### 2. Open in Browser

Navigate to: **http://localhost:8000**

## 🌐 Web Interface Features

### Dashboard Tab (📊)
- **System Status Indicator** - Shows if the system is online
- **Three Agent Cards**:
  - Monitor Agent - Collect market data
  - Planner Agent - Create investment plans
  - Predictor Agent - Validate portfolios
- **Quick Actions** - Run complete pipeline with one click

### Create Portfolio Tab (➕)
- **Customer Information Form**:
  - Customer ID
  - Risk Appetite (Low/Medium/High)
  - Investment Capital (₹)
  - Expected Return (%)
  - Investment Horizon (Short/Medium/Long Term)
  - Sector Preferences (checkboxes)
- **Submit Button** - Creates optimized investment plan
- **Results Display** - Shows portfolio summary with stocks, returns, and Sharpe ratio

### Market Data Tab (📈)
- **Refresh Button** - Loads latest market data
- **Stock Table** - Displays all stocks with:
  - Symbol, Company Name, Sector
  - Current Price
  - Price Change %
- **Market Overview** - Total stocks and last update time

### Results Tab (📋)
- **Customer ID Input** - Enter customer ID to load results
- **Investment Plan Display**:
  - Expected Return
  - Sharpe Ratio
  - Number of Stocks
- **Validation Results**:
  - Status (APPROVE/REVISE/REJECT)
  - Predicted Return
  - Confidence Level
  - Risk Assessment

## 🔌 API Endpoints

### System Status
```
GET /api/status
```
Returns system status and agent availability

### Monitor Agent
```
POST /api/run-monitor
```
Runs Monitor Agent to collect market data

### Planner Agent
```
POST /api/run-planner
Body: {
  "customer_id": "CUST_001",
  "risk_appetite": "MEDIUM",
  "investment_capital": 100000,
  "return_expectation": 15,
  "investment_horizon": "MEDIUM_TERM",
  "sector_preferences": ["TECH", "BANKING", "PHARMA"]
}
```
Creates optimized investment plan

### Predictor Agent
```
POST /api/run-predictor/{customer_id}
```
Validates investment plan for customer

### Get Market Data
```
GET /api/market-data
```
Returns latest market data from Monitor Agent

### Get Investment Plan
```
GET /api/investment-plan/{customer_id}
```
Returns investment plan for specific customer

### Get Validation
```
GET /api/validation/{customer_id}
```
Returns validation results for specific customer

### Run All Agents
```
POST /api/run-all
Body: {customer portfolio data}
```
Runs all three agents in sequence

## 🎨 Design Features

- **Gradient Background** - Purple gradient (667eea → 764ba2)
- **Responsive Layout** - Works on desktop and mobile
- **Animated Elements** - Pulsing status indicator, hover effects
- **Color-Coded Results**:
  - Success messages: Green
  - Error messages: Red
  - Loading states: Animated dots
- **Interactive Tables** - Hover effects on rows
- **Modern UI** - Rounded corners, shadows, smooth transitions

## 📊 Testing

Run the automated test suite:

```bash
python test_system.py
```

This will test all API endpoints and verify the complete pipeline.

## ✅ Current Status

- ✅ FastAPI backend server running
- ✅ Web interface accessible at http://localhost:8000
- ✅ Monitor Agent working (collects data for 35 stocks across 6 sectors)
- ✅ Existing data available for CUST_001
- ✅ All API endpoints functional
- ✅ Responsive design with modern UI

## 🔄 Complete Workflow

1. **Run Monitor Agent** → Collects market data for 35 stocks
2. **Create Portfolio** → Enter customer preferences and create plan
3. **Run Predictor** → Validate the investment plan
4. **View Results** → See complete analysis and recommendations

## 📝 Notes

- The system uses existing data from previous runs (CUST_001)
- Market data is stored in `data/monitor_data/`
- Investment plans are stored in `data/planner_data/`
- Validation results are stored in `data/predictor_data/`
- All data is in JSON format for easy API access

## 🎯 Next Steps (Optional)

- Add charts/graphs for portfolio visualization
- Implement real-time data updates
- Add user authentication
- Create PDF report generation
- Add historical performance tracking
- Implement portfolio comparison features

---

**Enjoy your Multi-Agent Investment System with Web Interface! 🚀**

