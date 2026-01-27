// CUDA Benchmark Report - Interactive Features
document.addEventListener('DOMContentLoaded', function() {
    // Navigation
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.report-section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Update active nav link
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            // Show corresponding section
            const targetId = this.getAttribute('href').substring(1);
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === targetId) {
                    section.classList.add('active');
                }
            });
            
            // Update URL hash
            history.pushState(null, '', '#' + targetId);
        });
    });
    
    // Sort buttons
    const sortButtons = document.querySelectorAll('.sort-btn');
    sortButtons.forEach(button => {
        button.addEventListener('click', function() {
            const sortType = this.getAttribute('data-sort');
            sortComparisonCards(sortType);
            
            // Update active sort button
            sortButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // Filter checkboxes
    const filterCheckboxes = document.querySelectorAll('.filter-checkbox');
    filterCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const type = this.getAttribute('data-type');
            const isChecked = this.checked;
            
            document.querySelectorAll(`.${type}-metric`).forEach(metric => {
                metric.style.display = isChecked ? 'block' : 'none';
            });
        });
    });
    
    document.querySelectorAll('.toggle-details-btn').forEach(button => {
        button.addEventListener('click', function() {
            const card = this.closest('.comparison-card');
            const details = card.querySelector('.metric-details');
            details.style.display = details.style.display === 'none' ? 'grid' : 'none';
            this.textContent = details.style.display === 'none' ? 'Show Details' : 'Hide Details';
        });
    });
    
    document.querySelectorAll('.details-header').forEach(header => {
        header.addEventListener('click', function() {
            const content = this.nextElementSibling.nextElementSibling;
            const icon = this.querySelector('i');
            
            if (content.style.display === 'none') {
                content.style.display = 'block';
                icon.className = 'fas fa-caret-down';
            } else {
                content.style.display = 'none';
                icon.className = 'fas fa-caret-right';
            }
        });
    });
    
    // Handle hash on page load
    if (window.location.hash) {
        const targetId = window.location.hash.substring(1);
        const targetLink = document.querySelector(`.nav-link[href="#${targetId}"]`);
        const targetSection = document.getElementById(targetId);
        
        if (targetLink && targetSection) {
            navLinks.forEach(l => l.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            
            targetLink.classList.add('active');
            targetSection.classList.add('active');
        }
    }
    
    // Sort function for comparison cards
    function sortComparisonCards(sortType) {
        const container = document.querySelector('.comparison-grid');
        const cards = Array.from(container.querySelectorAll('.comparison-card'));
        
        cards.sort((a, b) => {
            let aValue, bValue;
            
            switch(sortType) {
                case 'time':
                    aValue = parseFloat(a.getAttribute('data-time'));
                    bValue = parseFloat(b.getAttribute('data-time'));
                    return aValue - bValue;
                    
                case 'memory':
                    aValue = parseFloat(a.getAttribute('data-memory'));
                    bValue = parseFloat(b.getAttribute('data-memory'));
                    return aValue - bValue;
                    
                case 'name':
                    aValue = a.querySelector('h3').textContent.toLowerCase();
                    bValue = b.querySelector('h3').textContent.toLowerCase();
                    return aValue.localeCompare(bValue);
                    
                default:
                    return 0;
            }
        });
        
        // Reorder cards
        cards.forEach(card => container.appendChild(card));
    }
    
    // Highlight rows in summary table on hover
    document.querySelectorAll('.summary-row').forEach(row => {
        const benchmarkName = row.getAttribute('data-benchmark');
        
        row.addEventListener('mouseenter', function() {
            // You could add highlighting logic here
            this.style.backgroundColor = '#f0f7ff';
        });
        
        row.addEventListener('mouseleave', function() {
            this.style.backgroundColor = '';
        });
    });
});