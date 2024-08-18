document.addEventListener('DOMContentLoaded', () => {
    // Check if user is logged in
    const currentUser = JSON.parse(localStorage.getItem('currentUser'));
    if (!currentUser || currentUser.type !== 'investor') {
        window.location.href = 'login.html';
        return;
    }

    // Function to fetch and display matches
    async function fetchMatches() {
        const investorProfile = currentUser.profile; // Assuming the profile is stored with the user
        const response = await fetch('/match', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                investor_profile: investorProfile,
                target_profiles: [] // This should be filled with startup profiles from your database
            }),
        });

        if (response.ok) {
            const matches = await response.json();
            displayMatches(matches);
        } else {
            console.error('Failed to fetch matches');
        }
    }

    function displayMatches(matches) {
        // Implement this function to display matches in the UI
        console.log('Matches:', matches);
    }

    // Call fetchMatches when the page loads
    fetchMatches();

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