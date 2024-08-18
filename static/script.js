document.addEventListener('DOMContentLoaded', () => {
    const loginInvestor = document.getElementById('loginInvestor');
    const loginStartup = document.getElementById('loginStartup');
    const landingPage = document.getElementById('landingPage');
    const loginForm = document.getElementById('loginForm');
    const loginFormElement = document.getElementById('loginFormElement');
    const investorDashboard = document.getElementById('investorDashboard');
    const startupDashboard = document.getElementById('startupDashboard');

    let userType = '';

    function showSection(section) {
        landingPage.classList.remove('active');
        loginForm.classList.remove('active');
        investorDashboard.classList.remove('active');
        startupDashboard.classList.remove('active');
        section.classList.add('active');
    }

    loginInvestor.addEventListener('click', () => {
        userType = 'investor';
        showSection(loginForm);
    });

    loginStartup.addEventListener('click', () => {
        userType = 'startup';
        showSection(loginForm);
    });

    loginFormElement.addEventListener('submit', (e) => {
        e.preventDefault();
        // Login is now handled in login.js
    });

    async function loadInvestorDashboard() {
        const recommendations = document.getElementById('investorRecommendations');
        const portfolio = document.getElementById('investorPortfolio');
        const profile = document.getElementById('investorProfile');

        try {
            const response = await fetch('/investor_dashboard_data');
            if (response.ok) {
                const data = await response.json();
                recommendations.innerHTML = '<ul>' + data.recommendations.map(r => `<li>${r}</li>`).join('') + '</ul>';
                portfolio.innerHTML = '<ul>' + data.portfolio.map(p => `<li>${p.name} - $${p.amount}</li>`).join('') + '</ul>';
                profile.innerHTML = `<p>Name: ${data.profile.name}</p><p>Interests: ${data.profile.interests.join(', ')}</p>`;
            } else {
                throw new Error('Failed to fetch investor dashboard data');
            }
        } catch (error) {
            console.error('Error loading investor dashboard:', error);
            // Handle error (e.g., show an error message to the user)
        }
    }

    async function loadStartupDashboard() {
        const profile = document.getElementById('startupProfile');
        const funding = document.getElementById('startupFunding');
        const investors = document.getElementById('startupInvestors');
        const tips = document.getElementById('startupTips');

        try {
            const response = await fetch('/startup_dashboard_data');
            if (response.ok) {
                const data = await response.json();
                profile.innerHTML = `<p>Startup Name: ${data.profile.name}</p><p>Industry: ${data.profile.industry}</p>`;
                funding.innerHTML = `<p>Goal: $${data.funding.goal}</p><p>Raised: $${data.funding.raised}</p><progress value="${data.funding.percentage}" max="100"></progress>`;
                investors.innerHTML = '<ul>' + data.investors.map(i => `<li>${i.name} - ${i.status}</li>`).join('') + '</ul>';
                tips.innerHTML = '<ul>' + data.tips.map(t => `<li>${t}</li>`).join('') + '</ul>';
            } else {
                throw new Error('Failed to fetch startup dashboard data');
            }
        } catch (error) {
            console.error('Error loading startup dashboard:', error);
            // Handle error (e.g., show an error message to the user)
        }
    }

    // Check if user is logged in and load appropriate dashboard
    const currentUser = JSON.parse(localStorage.getItem('currentUser'));
    if (currentUser) {
        if (currentUser.type === 'investor') {
            showSection(investorDashboard);
            loadInvestorDashboard();
        } else if (currentUser.type === 'startup') {
            showSection(startupDashboard);
            loadStartupDashboard();
        }
    } else {
        showSection(landingPage);
    }
});