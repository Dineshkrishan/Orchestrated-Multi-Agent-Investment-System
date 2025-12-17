"""
Investment Planner Agent - Portfolio Strategy Developer
Multi-Agent Investment Management System

RESPONSIBILITY: Create optimized investment plans using MPT
Input: Customer portfolio data (from Manager) + Market data (from Monitor)
Output: Optimized portfolio allocation with risk-adjusted weights
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import os
import sys
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from logger.custom_logger import CustomLogger

# Configure custom logger
logger = CustomLogger().get_logger(__file__)


@dataclass
class CustomerPortfolio:
    """Customer portfolio specifications from Manager"""
    customer_id: str
    risk_appetite: str  # LOW, MEDIUM, HIGH
    investment_capital: float
    sector_preferences: List[str]
    return_expectation: float
    investment_horizon: str  # SHORT_TERM, MEDIUM_TERM, LONG_TERM
    max_single_stock_allocation: float = 0.15  # Default 15%
    min_diversification: int = 5  # Minimum number of stocks


@dataclass
class StockAllocation:
    """Individual stock allocation in the portfolio"""
    symbol: str
    company_name: str
    sector: str
    weight: float
    dollar_amount: float
    current_price: float
    shares: int
    expected_return: float
    risk_score: float
    selection_reason: str


@dataclass
class PortfolioPlan:
    """Complete investment plan output"""
    customer_id: str
    plan_date: datetime
    total_capital: float
    allocations: List[StockAllocation]
    portfolio_metrics: Dict
    risk_analysis: Dict
    diversification_score: float
    expected_annual_return: float
    sharpe_ratio: float
    max_drawdown_estimate: float
    rebalancing_recommendation: str


class InvestmentPlannerAgent:
    """Investment Planner Agent - Portfolio Optimization Engine"""
    
    def __init__(self):
        self.risk_profiles = {
            "LOW": {
                "volatility_threshold": 0.20,
                "max_single_allocation": 0.12,
                "min_stocks": 8,
                "sector_concentration": 0.30
            },
            "MEDIUM": {
                "volatility_threshold": 0.30,
                "max_single_allocation": 0.15,
                "min_stocks": 6,
                "sector_concentration": 0.40
            },
            "HIGH": {
                "volatility_threshold": 0.50,
                "max_single_allocation": 0.20,
                "min_stocks": 5,
                "sector_concentration": 0.50
            }
        }
        
        logger.info("Investment Planner Agent initialized")
    
    def load_market_data(self) -> Dict:
        """Load general market data from Monitor Agent"""
        try:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            market_data_path = os.path.join(project_root, "data", "monitor_data", "general_market_data.json")
            
            with open(market_data_path, 'r') as f:
                market_data = json.load(f)
            
            logger.info(f"Loaded market data: {len(market_data['stocks'])} stocks")
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to load market data: {e}")
            raise
    
    def filter_stocks_by_preferences(self, market_data: Dict, 
                                     customer: CustomerPortfolio) -> List[Dict]:
        """Filter stocks based on customer sector preferences and risk profile"""
        
        risk_config = self.risk_profiles[customer.risk_appetite]
        filtered_stocks = []
        
        for stock in market_data['stocks']:
            # Filter by sector preference
            if stock['sector'] not in customer.sector_preferences:
                continue
            
            # Filter by volatility (risk appetite)
            tech_indicators = stock.get('technical_indicators', {})
            volatility = tech_indicators.get('volatility', 0)
            
            if volatility > risk_config['volatility_threshold']:
                continue
            
            # Filter by signal strength (avoid strong downtrends)
            signal = tech_indicators.get('signal_strength', 'NEUTRAL')
            if signal in ['STRONG_DOWN', 'DOWN'] and customer.risk_appetite == 'LOW':
                continue
            
            # Filter by RSI (avoid overbought/oversold extremes for conservative investors)
            rsi = tech_indicators.get('rsi')
            if customer.risk_appetite == 'LOW' and rsi:
                if rsi > 75 or rsi < 25:
                    continue
            
            filtered_stocks.append(stock)
        
        logger.info(f"Filtered to {len(filtered_stocks)} stocks matching customer preferences")
        return filtered_stocks
    
    def calculate_expected_returns(self, stocks: List[Dict], 
                                   market_data: Dict) -> np.ndarray:
        """Calculate expected returns for each stock"""
        
        expected_returns = []
        
        for stock in stocks:
            tech = stock.get('technical_indicators', {})
            
            # Base return: recent price change
            price_change = stock.get('price_change_pct', 0) / 100
            
            # Momentum adjustment
            momentum = tech.get('momentum_20d', 0)
            
            # Signal strength adjustment
            signal = tech.get('signal_strength', 'NEUTRAL')
            signal_multiplier = {
                'STRONG_UP': 1.2,
                'UP': 1.1,
                'NEUTRAL': 1.0,
                'DOWN': 0.9,
                'STRONG_DOWN': 0.8
            }.get(signal, 1.0)
            
            # Sector performance adjustment
            sector_data = next((s for s in market_data['sectors'] 
                              if s['sector'] == stock['sector']), None)
            sector_performance = sector_data['performance'] / 100 if sector_data else 0
            
            # Combined expected return (annualized estimate)
            expected_return = (price_change * 0.3 + 
                             momentum * 0.4 + 
                             sector_performance * 0.3) * signal_multiplier
            
            # Annualize (rough estimate)
            expected_annual_return = expected_return * 12
            
            expected_returns.append(expected_annual_return)
        
        return np.array(expected_returns)
    
    def calculate_covariance_matrix(self, stocks: List[Dict]) -> np.ndarray:
        """Calculate covariance matrix from volatility data"""
        
        n = len(stocks)
        volatilities = []
        
        for stock in stocks:
            tech = stock.get('technical_indicators', {})
            volatility = tech.get('volatility', 0.20)
            volatilities.append(volatility)
        
        volatilities = np.array(volatilities)
        
        # Create correlation matrix (simplified - assume moderate correlation within sectors)
        correlation_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i+1, n):
                if stocks[i]['sector'] == stocks[j]['sector']:
                    correlation = 0.6  # Same sector
                else:
                    correlation = 0.3  # Different sector
                
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        # Convert to covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        return cov_matrix
    
    def optimize_portfolio_mpt(self, expected_returns: np.ndarray,
                               cov_matrix: np.ndarray,
                               customer: CustomerPortfolio,
                               stocks: List[Dict]) -> np.ndarray:
        """Optimize portfolio using Modern Portfolio Theory (Markowitz)"""
        
        n_assets = len(expected_returns)
        risk_config = self.risk_profiles[customer.risk_appetite]
        
        # Convert user's return expectation to decimal (e.g., 20% -> 0.20)
        target_return = customer.return_expectation / 100.0
        
        # Calculate max achievable return from available stocks
        max_achievable_return = np.max(expected_returns)
        
        # Objective function: Balance Sharpe Ratio and Target Return
        def hybrid_objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if portfolio_std == 0:
                return 1e10
            
            # Sharpe ratio component (negative for minimization)
            sharpe = portfolio_return / portfolio_std
            sharpe_component = -sharpe
            
            # Return target penalty (penalize deviation from target)
            return_deviation = abs(portfolio_return - target_return)
            return_penalty = return_deviation * 10  # Weight the penalty
            
            # Combined objective: balance risk-adjusted returns with target return
            # If target is achievable, prioritize hitting it
            # If target is too high, maximize Sharpe ratio
            if target_return <= max_achievable_return:
                # Target is achievable - balance both objectives
                objective = 0.4 * sharpe_component + 0.6 * return_penalty
            else:
                # Target too high - focus on Sharpe ratio but still consider target
                objective = 0.7 * sharpe_component + 0.3 * return_penalty
            
            return objective
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds for each weight
        max_weight = risk_config['max_single_allocation']
        bounds = tuple((0.0, max_weight) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            hybrid_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            logger.warning("Optimization did not converge, using equal weights")
            weights = initial_weights
        else:
            weights = result.x
        
        # Filter out very small weights (< 1%)
        weights[weights < 0.01] = 0
        
        # Renormalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = initial_weights
        
        # Ensure minimum diversification
        non_zero_weights = np.sum(weights > 0)
        if non_zero_weights < risk_config['min_stocks']:
            # Force more diversification
            top_indices = np.argsort(expected_returns)[-risk_config['min_stocks']:]
            weights = np.zeros(n_assets)
            weights[top_indices] = 1.0 / len(top_indices)
        
        return weights

    
    def apply_sector_constraints(self, weights: np.ndarray, 
                                stocks: List[Dict],
                                customer: CustomerPortfolio) -> np.ndarray:
        """Apply sector concentration constraints"""
        
        risk_config = self.risk_profiles[customer.risk_appetite]
        max_sector_weight = risk_config['sector_concentration']
        
        # Calculate current sector allocations
        sector_weights = {}
        for i, stock in enumerate(stocks):
            sector = stock['sector']
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += weights[i]
        
        # Adjust if any sector exceeds limit
        adjusted_weights = weights.copy()
        
        for sector, sector_weight in sector_weights.items():
            if sector_weight > max_sector_weight:
                # Scale down this sector's stocks
                sector_indices = [i for i, s in enumerate(stocks) if s['sector'] == sector]
                scale_factor = max_sector_weight / sector_weight
                
                for idx in sector_indices:
                    adjusted_weights[idx] *= scale_factor
        
        # Renormalize
        if adjusted_weights.sum() > 0:
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        return adjusted_weights
    
    def create_investment_plan(self, customer: CustomerPortfolio) -> PortfolioPlan:
        """Create complete investment plan for customer"""
        
        logger.info(f"Creating investment plan for customer: {customer.customer_id}")
        
        # Load market data
        market_data = self.load_market_data()
        
        # Filter stocks by customer preferences
        suitable_stocks = self.filter_stocks_by_preferences(market_data, customer)
        
        if len(suitable_stocks) < customer.min_diversification:
            raise ValueError(f"Insufficient stocks ({len(suitable_stocks)}) for diversification")
        
        # Calculate expected returns and covariance
        expected_returns = self.calculate_expected_returns(suitable_stocks, market_data)
        cov_matrix = self.calculate_covariance_matrix(suitable_stocks)
        
        # Optimize portfolio
        optimal_weights = self.optimize_portfolio_mpt(
            expected_returns, cov_matrix, customer, suitable_stocks
        )
        
        # Apply sector constraints
        final_weights = self.apply_sector_constraints(
            optimal_weights, suitable_stocks, customer
        )
        
        # Create stock allocations
        allocations = []
        
        for i, stock in enumerate(suitable_stocks):
            if final_weights[i] < 0.01:  # Skip negligible allocations
                continue
            
            weight = final_weights[i]
            dollar_amount = customer.investment_capital * weight
            current_price = stock['current_price']
            shares = int(dollar_amount / current_price)
            actual_amount = shares * current_price
            
            # Selection reason
            tech = stock.get('technical_indicators', {})
            signal = tech.get('signal_strength', 'NEUTRAL')
            rsi = tech.get('rsi', 50)
            
            reasons = []
            if signal in ['STRONG_UP', 'UP']:
                reasons.append(f"Positive momentum ({signal})")
            if 30 <= rsi <= 70:
                reasons.append(f"Healthy RSI ({rsi:.1f})")
            if tech.get('momentum_20d', 0) > 0:
                reasons.append("Upward trend")
            
            selection_reason = "; ".join(reasons) if reasons else "Sector diversification"
            
            allocation = StockAllocation(
                symbol=stock['symbol'],
                company_name=stock['company_name'],
                sector=stock['sector'],
                weight=weight,
                dollar_amount=actual_amount,
                current_price=current_price,
                shares=shares,
                expected_return=expected_returns[i],
                risk_score=tech.get('volatility', 0),
                selection_reason=selection_reason
            )
            
            allocations.append(allocation)
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(final_weights, expected_returns)
        portfolio_variance = np.dot(final_weights.T, np.dot(cov_matrix, final_weights))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0
        
        # Diversification score (higher is better)
        non_zero_weights = final_weights[final_weights > 0]
        diversification_score = len(non_zero_weights) / len(suitable_stocks)
        
        # Calculate sector distribution
        sector_distribution = {}
        for allocation in allocations:
            sector = allocation.sector
            if sector not in sector_distribution:
                sector_distribution[sector] = 0
            sector_distribution[sector] += allocation.weight
        
        # Risk analysis
        risk_analysis = {
            "portfolio_volatility": portfolio_std,
            "risk_level": customer.risk_appetite,
            "max_drawdown_estimate": -2 * portfolio_std,  # Rough estimate
            "sector_distribution": sector_distribution,
            "concentration_risk": max(final_weights) if len(final_weights) > 0 else 0
        }
        
        # Portfolio metrics
        portfolio_metrics = {
            "number_of_stocks": len(allocations),
            "total_invested": sum(a.dollar_amount for a in allocations),
            "cash_remaining": customer.investment_capital - sum(a.dollar_amount for a in allocations),
            "average_allocation": np.mean([a.weight for a in allocations]),
            "max_allocation": max([a.weight for a in allocations]) if allocations else 0,
            "min_allocation": min([a.weight for a in allocations]) if allocations else 0
        }
        
        # Rebalancing recommendation
        if customer.investment_horizon == "SHORT_TERM":
            rebalancing = "Review monthly"
        elif customer.investment_horizon == "MEDIUM_TERM":
            rebalancing = "Review quarterly"
        else:
            rebalancing = "Review semi-annually"
        
        # Create portfolio plan
        plan = PortfolioPlan(
            customer_id=customer.customer_id,
            plan_date=datetime.now(),
            total_capital=customer.investment_capital,
            allocations=allocations,
            portfolio_metrics=portfolio_metrics,
            risk_analysis=risk_analysis,
            diversification_score=diversification_score,
            expected_annual_return=portfolio_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_estimate=risk_analysis["max_drawdown_estimate"],
            rebalancing_recommendation=rebalancing
        )
        
        logger.info(f"Investment plan created: {len(allocations)} stocks, "
                   f"Expected return: {portfolio_return:.2%}")
        
        return plan
    
    def export_plan(self, plan: PortfolioPlan):
        """Export investment plan to JSON for Target Predictor"""
        
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        export_dir = os.path.join(project_root, "data", "planner_data")
        os.makedirs(export_dir, exist_ok=True)
        
        plan_data = {
            "customer_id": plan.customer_id,
            "plan_date": plan.plan_date.isoformat(),
            "total_capital": plan.total_capital,
            "allocations": [
                {
                    "symbol": a.symbol,
                    "company_name": a.company_name,
                    "sector": a.sector,
                    "weight": a.weight,
                    "dollar_amount": a.dollar_amount,
                    "shares": a.shares,
                    "expected_return": a.expected_return,
                    "risk_score": a.risk_score
                }
                for a in plan.allocations
            ],
            "portfolio_metrics": plan.portfolio_metrics,
            "risk_analysis": plan.risk_analysis,
            "expected_annual_return": plan.expected_annual_return,
            "sharpe_ratio": plan.sharpe_ratio,
            "diversification_score": plan.diversification_score
        }
        
        filename = f"{export_dir}/investment_plan_{plan.customer_id}.json"
        with open(filename, 'w') as f:
            json.dump(plan_data, f, indent=2)
        
        logger.info(f"Investment plan exported: {filename}")
        return filename
    
    def print_plan_summary(self, plan: PortfolioPlan):
        """Print readable plan summary"""
        
        try:
            print("\n" + "="*80)
            print(f"📊 INVESTMENT PLAN SUMMARY - {plan.customer_id}")
            print("="*80)
            
            print(f"\n💰 Capital Allocation:")
            print(f"   Total Capital: ₹{plan.total_capital:,.2f}")
            print(f"   Invested: ₹{plan.portfolio_metrics['total_invested']:,.2f}")
            print(f"   Cash Remaining: ₹{plan.portfolio_metrics['cash_remaining']:,.2f}")
            
            print(f"\n📈 Portfolio Metrics:")
            print(f"   Expected Annual Return: {plan.expected_annual_return:.2%}")
            print(f"   Sharpe Ratio: {plan.sharpe_ratio:.3f}")
            print(f"   Diversification Score: {plan.diversification_score:.2%}")
            print(f"   Number of Stocks: {plan.portfolio_metrics['number_of_stocks']}")
            
            print(f"\n⚠️ Risk Analysis:")
            print(f"   Portfolio Volatility: {plan.risk_analysis['portfolio_volatility']:.2%}")
            print(f"   Max Drawdown Estimate: {plan.max_drawdown_estimate:.2%}")
            print(f"   Concentration Risk: {plan.risk_analysis['concentration_risk']:.2%}")
            
            print(f"\n🏭 Sector Distribution:")
            for sector, weight in plan.risk_analysis['sector_distribution'].items():
                print(f"   {sector}: {weight:.2%}")
            
            print(f"\n💼 Stock Allocations:")
            for allocation in sorted(plan.allocations, key=lambda x: x.weight, reverse=True):
                print(f"   {allocation.symbol} ({allocation.sector})")
                print(f"      Weight: {allocation.weight:.2%} | Amount: ₹{allocation.dollar_amount:,.2f}")
                print(f"      Shares: {allocation.shares} @ ₹{allocation.current_price:.2f}")
                print(f"      Expected Return: {allocation.expected_return:.2%}")
                print(f"      Reason: {allocation.selection_reason}")
            
            print(f"\n🔄 Rebalancing: {plan.rebalancing_recommendation}")
            print("="*80)
        except UnicodeEncodeError:
            # Fallback to ASCII-only output if encoding fails
            print("\n" + "="*80)
            print(f"INVESTMENT PLAN SUMMARY - {plan.customer_id}")
            print("="*80)
            
            print(f"\nCapital Allocation:")
            print(f"   Total Capital: Rs.{plan.total_capital:,.2f}")
            print(f"   Invested: Rs.{plan.portfolio_metrics['total_invested']:,.2f}")
            print(f"   Cash Remaining: Rs.{plan.portfolio_metrics['cash_remaining']:,.2f}")
            
            print(f"\nPortfolio Metrics:")
            print(f"   Expected Annual Return: {plan.expected_annual_return:.2%}")
            print(f"   Sharpe Ratio: {plan.sharpe_ratio:.3f}")
            print(f"   Diversification Score: {plan.diversification_score:.2%}")
            print(f"   Number of Stocks: {plan.portfolio_metrics['number_of_stocks']}")
            
            print(f"\nRisk Analysis:")
            print(f"   Portfolio Volatility: {plan.risk_analysis['portfolio_volatility']:.2%}")
            print(f"   Max Drawdown Estimate: {plan.max_drawdown_estimate:.2%}")
            print(f"   Concentration Risk: {plan.risk_analysis['concentration_risk']:.2%}")
            
            print(f"\nSector Distribution:")
            for sector, weight in plan.risk_analysis['sector_distribution'].items():
                print(f"   {sector}: {weight:.2%}")
            
            print(f"\nStock Allocations:")
            for allocation in sorted(plan.allocations, key=lambda x: x.weight, reverse=True):
                print(f"   {allocation.symbol} ({allocation.sector})")
                print(f"      Weight: {allocation.weight:.2%} | Amount: Rs.{allocation.dollar_amount:,.2f}")
                print(f"      Shares: {allocation.shares} @ Rs.{allocation.current_price:.2f}")
                print(f"      Expected Return: {allocation.expected_return:.2%}")
                print(f"      Reason: {allocation.selection_reason}")
            
            print(f"\nRebalancing: {plan.rebalancing_recommendation}")
            print("="*80)



def main():
    """Run the Investment Planner Agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Investment Planner Agent')
    parser.add_argument('--customer_id', type=str, default="CUST_001", help='Customer ID')
    parser.add_argument('--risk_appetite', type=str, default="MEDIUM", help='Risk Appetite (LOW, MEDIUM, HIGH)')
    parser.add_argument('--investment_capital', type=float, default=100000.0, help='Investment Capital')
    parser.add_argument('--sector_preferences', type=str, default="TECH,PHARMA,BANKING", help='Comma-separated sector preferences')
    parser.add_argument('--return_expectation', type=float, default=12.0, help='Expected Return (%)')
    parser.add_argument('--investment_horizon', type=str, default="MEDIUM_TERM", help='Investment Horizon')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Investment Planner Agent for {args.customer_id}")
    
    # Parse sector preferences
    sectors = [s.strip() for s in args.sector_preferences.split(',')]
    
    # Create customer portfolio from arguments
    customer = CustomerPortfolio(
        customer_id=args.customer_id,
        risk_appetite=args.risk_appetite,
        investment_capital=args.investment_capital,
        sector_preferences=sectors,
        return_expectation=args.return_expectation,
        investment_horizon=args.investment_horizon
    )
    
    # Create planner
    planner = InvestmentPlannerAgent()
    
    try:
        # Generate investment plan
        plan = planner.create_investment_plan(customer)
        
        # Print summary
        planner.print_plan_summary(plan)
        
        # Export for Target Predictor
        export_file = planner.export_plan(plan)
        
        print(f"\n✅ Plan exported to: {export_file}")
        
    except Exception as e:
        logger.error(f"Error creating investment plan: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()