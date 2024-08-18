document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    const emailInput = document.querySelector('#email');

    form.addEventListener('submit', function(e) {
        if (!emailInput.value.trim()) {
            e.preventDefault();
            alert('Please enter your email address.');
        }
    });
});