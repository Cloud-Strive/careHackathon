document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('investorProfileForm');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const profileData = Object.fromEntries(formData);

        try {
            const response = await fetch('/add_profile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    profile_type: 'investor',
                    profile_data: profileData
                }),
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Profile creation result:', result);
                alert('Profile created successfully!');
                window.location.href = 'investor-dashboard.html';
            } else {
                throw new Error('Failed to create profile');
            }
        } catch (error) {
            console.error('Error creating profile:', error);
            alert('Failed to create profile. Please try again.');
        }
    });
});