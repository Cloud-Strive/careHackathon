document.addEventListener('DOMContentLoaded', () => {
    // Check if user is logged in
    const currentUser = JSON.parse(localStorage.getItem('currentUser'));
    if (!currentUser || currentUser.type !== 'startup') {
        window.location.href = 'login.html';
        return;
    }

    // Function to fetch and display potential investors
    async function fetchPotentialInvestors() {
        const startupProfile = currentUser.profile; // Assuming the profile is stored with the user
        const response = await fetch('/match', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                investor_profile: '', // This should be an empty string for startups
                target_profiles: [startupProfile]
            }),
        });

        if (response.ok) {
            const matches = await response.json();
            displayPotentialInvestors(matches);
        } else {
            console.error('Failed to fetch potential investors');
        }
    }

    function displayPotentialInvestors(matches) {
        // Implement this function to display potential investors in the UI
        console.log('Potential Investors:', matches);
    }

    // Call fetchPotentialInvestors when the page loads
    fetchPotentialInvestors();

    // Update logout functionality
    function setupLogout() {
        const logoutButton = document.getElementById('logout');
        logoutButton.addEventListener('click', (e) => {
            e.preventDefault();
            localStorage.removeItem('currentUser');
            window.location.href = 'login.html';
        });
    }

    setupLogout();
});