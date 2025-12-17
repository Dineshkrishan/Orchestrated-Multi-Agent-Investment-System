// API Base URL
const API_BASE = '';

// Check system status on load
window.addEventListener('DOMContentLoaded', () => {
    checkStatus();

    // Setup form submission
    document.getElementById('portfolioForm').addEventListener('submit', handleFormSubmit);
});

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });

    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show selected tab
    document.getElementById(tabName).classList.add('active');

    // Add active class to clicked button
    event.target.classList.add('active');
}

// Check system status
async function checkStatus() {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const statusDot = document.getElementById('statusDot');

    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();

        statusText.textContent = 'System Online';
        statusDot.style.background = 'hsl(145, 70%, 50%)';
        statusDot.style.boxShadow = '0 0 10px hsl(145, 70%, 50%)';

        // Add success animation
        statusIndicator.style.animation = 'none';
        setTimeout(() => {
            statusIndicator.style.animation = '';
        }, 10);
    } catch (error) {
        statusText.textContent = 'System Offline';
        statusDot.style.background = 'hsl(0, 75%, 60%)';
        statusDot.style.boxShadow = '0 0 10px hsl(0, 75%, 60%)';
    }
}

// Run Monitor Agent
async function runMonitor() {
    const statusDiv = document.getElementById('monitorStatus');
    statusDiv.innerHTML = '<div class="loading">Running Monitor Agent</div>';

    try {
        const response = await fetch(`${API_BASE}/api/run-monitor`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.status === 'success') {
            statusDiv.innerHTML = `
                <div class="success-message">
                    ✅ ${data.message}<br>
                    Stocks: ${data.data?.stocks_count || 'N/A'} | 
                    Sectors: ${data.data?.sectors_count || 'N/A'}
                </div>
            `;
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        statusDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
    }
}

// Run Predictor Agent
async function runPredictor() {
    const customerId = prompt('Enter Customer ID:', 'CUST_001');
    if (!customerId) return;

    const statusDiv = document.getElementById('predictorStatus');
    statusDiv.innerHTML = '<div class="loading">Running Predictor Agent</div>';

    try {
        const response = await fetch(`${API_BASE}/api/run-predictor/${customerId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.status === 'success') {
            statusDiv.innerHTML = `
                <div class="success-message">
                    ✅ ${data.message}<br>
                    Decision: ${data.data?.status || 'N/A'}
                </div>
            `;
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        statusDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
    }
}

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();

    const sectors = Array.from(document.querySelectorAll('input[name="sector"]:checked'))
        .map(cb => cb.value);

    const portfolio = {
        customer_id: document.getElementById('customerId').value,
        risk_appetite: document.getElementById('riskAppetite').value,
        investment_capital: parseFloat(document.getElementById('investmentCapital').value),
        return_expectation: parseFloat(document.getElementById('returnExpectation').value),
        investment_horizon: document.getElementById('investmentHorizon').value,
        sector_preferences: sectors
    };

    const resultDiv = document.getElementById('planResult');
    resultDiv.innerHTML = '<div class="loading">Creating investment plan</div>';

    try {
        const response = await fetch(`${API_BASE}/api/run-planner`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(portfolio)
        });

        const data = await response.json();

        if (data.status === 'success') {
            displayPlanResults(data.data, resultDiv);
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        resultDiv.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
    }
}

// Display plan results
function displayPlanResults(plan, container) {
    if (!plan) {
        container.innerHTML = '<div class="error-message">No plan data available</div>';
        return;
    }

    let stocksHtml = '';
    if (plan.allocations && plan.allocations.length > 0) {
        stocksHtml = `
            <table class="stock-table">
                <thead>
                    <tr>
                        <th>Stock</th>
                        <th>Sector</th>
                        <th>Weight</th>
                        <th>Amount</th>
                        <th>Expected Return</th>
                    </tr>
                </thead>
                <tbody>
                    ${plan.allocations.map(stock => `
                        <tr>
                            <td>${stock.symbol}</td>
                            <td>${stock.sector}</td>
                            <td>${(stock.weight * 100).toFixed(2)}%</td>
                            <td>₹${stock.dollar_amount?.toLocaleString()}</td>
                            <td>${(stock.expected_return * 100).toFixed(2)}%</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    }

    const html = `
        <div class="success-message">
            ✅ Investment plan created successfully!
        </div>

        <div class="metric-card">
            <h4>Portfolio Summary</h4>
            <p><strong>Customer ID:</strong> ${plan.customer_id}</p>
            <p><strong>Total Capital:</strong> ₹${plan.total_capital?.toLocaleString()}</p>
            <p><strong>Invested Amount:</strong> ₹${plan.portfolio_metrics?.total_invested?.toLocaleString()}</p>
            <p><strong>Expected Return:</strong> ${formatNumber(plan.expected_annual_return, 2, true)}</p>
            <p><strong>Sharpe Ratio:</strong> ${formatNumber(plan.sharpe_ratio, 3)}</p>
        </div>

        ${stocksHtml}
    `;

    container.innerHTML = html;
}

// Run all agents
async function runAllAgents() {
    if (!confirm('This will run all three agents in sequence. Continue?')) return;

    const sectors = ['TECH', 'BANKING', 'PHARMA'];

    const portfolio = {
        customer_id: 'CUST_001',
        risk_appetite: 'MEDIUM',
        investment_capital: 100000,
        return_expectation: 15,
        investment_horizon: 'MEDIUM_TERM',
        sector_preferences: sectors
    };

    // Show loading in all status divs
    document.getElementById('monitorStatus').innerHTML = '<div class="loading">Running</div>';
    document.getElementById('plannerStatus').innerHTML = '<div class="loading">Waiting</div>';
    document.getElementById('predictorStatus').innerHTML = '<div class="loading">Waiting</div>';

    try {
        const response = await fetch(`${API_BASE}/api/run-all`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(portfolio)
        });

        const data = await response.json();

        if (data.status === 'success') {
            document.getElementById('monitorStatus').innerHTML =
                '<div class="success-message">✅ Completed</div>';
            document.getElementById('plannerStatus').innerHTML =
                '<div class="success-message">✅ Completed</div>';
            document.getElementById('predictorStatus').innerHTML =
                '<div class="success-message">✅ Completed</div>';

            alert('All agents completed successfully! Check the Results tab.');
        } else {
            throw new Error(data.message);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// Load market data
async function loadMarketData() {
    const container = document.getElementById('marketDataContainer');
    container.innerHTML = '<div class="loading">Loading market data</div>';

    try {
        const response = await fetch(`${API_BASE}/api/market-data`);
        const data = await response.json();

        if (data.stocks && data.stocks.length > 0) {
            const html = `
                <h3>Market Overview</h3>
                <p><strong>Total Stocks:</strong> ${data.stocks.length}</p>
                <p><strong>Last Updated:</strong> ${data.timestamp}</p>

                <table class="stock-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Company</th>
                            <th>Sector</th>
                            <th>Price</th>
                            <th>Change %</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.stocks.slice(0, 20).map(stock => `
                            <tr>
                                <td>${stock.symbol}</td>
                                <td>${stock.company_name}</td>
                                <td>${stock.sector}</td>
                                <td>₹${stock.current_price?.toFixed(2)}</td>
                                <td style="color: ${stock.price_change_pct >= 0 ? 'green' : 'red'}">
                                    ${stock.price_change_pct?.toFixed(2)}%
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
            container.innerHTML = html;
        } else {
            container.innerHTML = '<div class="error-message">No market data available. Run Monitor Agent first.</div>';
        }
    } catch (error) {
        container.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
    }
}

// Helper to safely format numbers
const formatNumber = (val, decimals = 2, isPercent = false) => {
    if (val === null || val === undefined || isNaN(val)) return 'N/A';
    const num = isPercent ? val * 100 : val;
    return num.toFixed(decimals) + (isPercent ? '%' : '');
};

// Load results
async function loadResults() {
    const customerId = document.getElementById('resultCustomerId').value;
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '<div class="loading">Loading results</div>';

    try {
        // Load investment plan
        const planResponse = await fetch(`${API_BASE}/api/investment-plan/${customerId}`);
        const plan = await planResponse.json();

        // Load validation
        const validationResponse = await fetch(`${API_BASE}/api/validation/${customerId}`);
        const validation = await validationResponse.json();

        const html = `
            <div class="metric-card">
                <h4>Investment Plan</h4>
                <p><strong>Expected Return:</strong> ${formatNumber(plan.expected_annual_return, 2, true)}</p>
                <p><strong>Sharpe Ratio:</strong> ${formatNumber(plan.sharpe_ratio, 3)}</p>
                <p><strong>Number of Stocks:</strong> ${plan.allocations?.length}</p>
            </div>

            <div class="metric-card">
                <h4>Validation Results</h4>
                <p><strong>Status:</strong> ${validation.status}</p>
                <p><strong>Predicted Return:</strong> ${formatNumber(validation.predicted_return, 2, true)}</p>
                <p><strong>Confidence:</strong> ${formatNumber(validation.confidence_level, 1, true)}</p>
                <p><strong>Risk Acceptable:</strong> ${validation.risk_analysis?.risk_acceptable ? 'Yes' : 'No'}</p>
            </div>
        `;

        container.innerHTML = html;
    } catch (error) {
        container.innerHTML = `<div class="error-message">❌ Error: ${error.message}</div>`;
    }
}

