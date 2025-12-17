import subprocess
import json
import os
import sys

def test_planner_integration():
    """Test if planner agent accepts arguments correctly"""
    
    # Define test cases
    test_cases = [
        {
            "customer_id": "TEST_001",
            "risk_appetite": "HIGH",
            "investment_capital": 500000.0,
            "sector_preferences": "TECH,PHARMA",
            "return_expectation": 20.0,
            "investment_horizon": "LONG_TERM"
        },
        {
            "customer_id": "TEST_002",
            "risk_appetite": "LOW",
            "investment_capital": 10000.0,
            "sector_preferences": "BANKING",
            "return_expectation": 5.0,
            "investment_horizon": "SHORT_TERM"
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting with Customer ID: {case['customer_id']}")
        
        # Construct command
        cmd = [
            "python", "src/planner/planner_agent.py",
            "--customer_id", case['customer_id'],
            "--risk_appetite", case['risk_appetite'],
            "--investment_capital", str(case['investment_capital']),
            "--sector_preferences", case['sector_preferences'],
            "--return_expectation", str(case['return_expectation']),
            "--investment_horizon", case['investment_horizon']
        ]
        
        # Run agent
        try:
            # Set environment to force UTF-8
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=60,
                cwd=os.getcwd(),
                env=env
            )
            
            if result.returncode != 0:
                print(f"❌ Agent failed with return code {result.returncode}")
                print(f"Stderr: {result.stderr}")
                continue
                
            # Check output file
            output_file = f"data/planner_data/investment_plan_{case['customer_id']}.json"
            if not os.path.exists(output_file):
                print(f"❌ Output file not found: {output_file}")
                continue
                
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Verify data matches input
            if data['customer_id'] != case['customer_id']:
                print(f"❌ Customer ID mismatch: Expected {case['customer_id']}, got {data['customer_id']}")
            elif data['total_capital'] != case['investment_capital']:
                print(f"❌ Capital mismatch: Expected {case['investment_capital']}, got {data['total_capital']}")
            else:
                print(f"✅ Test passed for {case['customer_id']}")
                print(f"   Capital: {data['total_capital']}")
                print(f"   Risk Analysis: {data['risk_analysis']['risk_level']}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_planner_integration()
