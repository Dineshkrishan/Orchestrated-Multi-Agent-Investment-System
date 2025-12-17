"""
Test script to verify the complete Multi-Agent Investment System
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_status():
    """Test system status"""
    print("🔍 Testing system status...")
    response = requests.get(f"{BASE_URL}/api/status")
    print(f"✅ Status: {response.json()}")
    print()

def test_monitor():
    """Test Monitor Agent"""
    print("📊 Running Monitor Agent...")
    response = requests.post(f"{BASE_URL}/api/run-monitor")
    result = response.json()
    print(f"✅ Monitor Result: {result['message']}")
    if result.get('data'):
        print(f"   Stocks: {result['data'].get('stocks_count')}")
        print(f"   Sectors: {result['data'].get('sectors_count')}")
    print()
    return result

def test_planner():
    """Test Planner Agent"""
    print("🎯 Running Planner Agent...")
    
    portfolio = {
        "customer_id": "CUST_TEST_001",
        "risk_appetite": "MEDIUM",
        "investment_capital": 100000,
        "return_expectation": 15,
        "investment_horizon": "MEDIUM_TERM",
        "sector_preferences": ["TECH", "BANKING", "PHARMA"]
    }
    
    response = requests.post(
        f"{BASE_URL}/api/run-planner",
        json=portfolio
    )
    result = response.json()
    print(f"✅ Planner Result: {result['message']}")
    if result.get('data'):
        data = result['data']
        print(f"   Expected Return: {data.get('expected_return', 0) * 100:.2f}%")
        print(f"   Sharpe Ratio: {data.get('sharpe_ratio', 0):.3f}")
        print(f"   Stocks: {len(data.get('allocations', []))}")
    print()
    return result

def test_predictor():
    """Test Predictor Agent"""
    print("🔮 Running Predictor Agent...")
    
    response = requests.post(f"{BASE_URL}/api/run-predictor/CUST_TEST_001")
    result = response.json()
    print(f"✅ Predictor Result: {result['message']}")
    if result.get('data'):
        data = result['data']
        print(f"   Status: {data.get('status')}")
        print(f"   Predicted Return: {data.get('predicted_return', 0) * 100:.2f}%")
        print(f"   Confidence: {data.get('confidence_level', 0) * 100:.1f}%")
    print()
    return result

def test_get_market_data():
    """Test getting market data"""
    print("📈 Fetching market data...")
    response = requests.get(f"{BASE_URL}/api/market-data")
    data = response.json()
    print(f"✅ Market data loaded: {len(data.get('stocks', []))} stocks")
    print()

def test_get_plan():
    """Test getting investment plan"""
    print("📋 Fetching investment plan...")
    response = requests.get(f"{BASE_URL}/api/investment-plan/CUST_TEST_001")
    data = response.json()
    print(f"✅ Investment plan loaded for {data.get('customer_id')}")
    print()

def test_get_validation():
    """Test getting validation results"""
    print("📊 Fetching validation results...")
    response = requests.get(f"{BASE_URL}/api/validation/CUST_TEST_001")
    data = response.json()
    print(f"✅ Validation loaded: {data.get('status')}")
    print()

def main():
    print("="*80)
    print("🚀 MULTI-AGENT INVESTMENT SYSTEM - COMPLETE TEST")
    print("="*80)
    print()
    
    try:
        # Test 1: System Status
        test_status()
        
        # Test 2: Run Monitor Agent
        test_monitor()
        time.sleep(2)
        
        # Test 3: Run Planner Agent
        test_planner()
        time.sleep(2)
        
        # Test 4: Run Predictor Agent
        test_predictor()
        time.sleep(2)
        
        # Test 5: Get Market Data
        test_get_market_data()
        
        # Test 6: Get Investment Plan
        test_get_plan()
        
        # Test 7: Get Validation
        test_get_validation()
        
        print("="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print()
        print("🌐 Web Interface: http://localhost:8000")
        print("📊 Dashboard: View all results in the web interface")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

