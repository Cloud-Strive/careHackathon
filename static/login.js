document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginFormElement');
    const loginMessage = document.getElementById('loginMessage');

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password }),
            });

            if (response.ok) {
                const user = await response.json();
                localStorage.setItem('currentUser', JSON.stringify(user));
                loginMessage.textContent = 'Login successful. Redirecting...';
                
                setTimeout(() => {
                    if (user.type === 'investor') {
                        window.location.href = 'investor-dashboard.html';
                    } else if (user.type === 'startup') {
                        window.location.href = 'startup-dashboard.html';
                    }
                }, 1500);
            } else {
                loginMessage.textContent = 'Invalid email or password. Please try again.';
            }
        } catch (error) {
            console.error('Login error:', error);
            loginMessage.textContent = 'An error occurred. Please try again later.';
        }
    });
});