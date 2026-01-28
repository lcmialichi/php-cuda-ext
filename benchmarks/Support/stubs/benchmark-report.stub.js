// CUDA Benchmark Report - Interactive Features
document.addEventListener('DOMContentLoaded', function () {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.report-section');

    document.addEventListener('click', function (e) {
        if (e.target.closest('.toggle-group-btn') || e.target.closest('.group-header')) {
            const btn = e.target.closest('.toggle-group-btn') ||
                e.target.closest('.group-header').querySelector('.toggle-group-btn');
            const groupBody = btn.closest('.test-group').querySelector('.group-body');
            const icon = btn.querySelector('i');

            groupBody.classList.toggle('expanded');
            btn.classList.toggle('rotated');
            icon.className = groupBody.classList.contains('expanded') ?
                'fas fa-chevron-up' : 'fas fa-chevron-down';
        }

        if (e.target.closest('.toggle-run-details-btn')) {
            const btn = e.target.closest('.toggle-run-details-btn');
            const runId = btn.getAttribute('data-run');
            const details = document.getElementById(runId);

            details.classList.toggle('expanded');
            btn.innerHTML = details.classList.contains('expanded') ?
                '<i class="fas fa-chart-bar"></i> Hide Details' :
                '<i class="fas fa-chart-bar"></i> View Details';
        }
    });

    document.addEventListener('change', function (e) {
        if (e.target.classList.contains('sort-select')) {
            const sortBy = e.target.value;
            sortTestGroups(sortBy);
        }
    });

    function sortTestGroups(sortBy) {
        const container = document.querySelector('.comparison-groups');
        const groups = Array.from(container.querySelectorAll('.test-group'));

        groups.sort((a, b) => {
            let aValue, bValue;

            switch (sortBy) {
                case 'name':
                    aValue = a.getAttribute('data-name').toLowerCase();
                    bValue = b.getAttribute('data-name').toLowerCase();
                    return aValue.localeCompare(bValue);

                case 'time':
                    aValue = parseFloat(a.getAttribute('data-time'));
                    bValue = parseFloat(b.getAttribute('data-time'));
                    return aValue - bValue;

                case 'memory':
                    aValue = parseFloat(a.getAttribute('data-memory'));
                    bValue = parseFloat(b.getAttribute('data-memory'));
                    return aValue - bValue;

                case 'runs':
                    aValue = parseInt(a.getAttribute('data-runs'));
                    bValue = parseInt(b.getAttribute('data-runs'));
                    return bValue - aValue; // Mais runs primeiro

                default:
                    return 0;
            }
        });

        groups.forEach(group => container.appendChild(group));
    }

    document.addEventListener('DOMContentLoaded', function () {
        const controls = document.querySelector('.controls');
        if (controls) {
            const expandAllBtn = document.createElement('button');
            expandAllBtn.className = 'expand-all-btn';
            expandAllBtn.innerHTML = '<i class="fas fa-expand-alt"></i> Expand All';
            expandAllBtn.style.marginLeft = '1rem';
            controls.querySelector('.sort-controls').appendChild(expandAllBtn);

            expandAllBtn.addEventListener('click', function () {
                const groups = document.querySelectorAll('.test-group');
                const allExpanded = Array.from(groups).every(group =>
                    group.querySelector('.group-body').classList.contains('expanded'));

                groups.forEach(group => {
                    const body = group.querySelector('.group-body');
                    const btn = group.querySelector('.toggle-group-btn');
                    const icon = btn.querySelector('i');

                    if (allExpanded) {
                        body.classList.remove('expanded');
                        btn.classList.remove('rotated');
                        icon.className = 'fas fa-chevron-down';
                    } else {
                        body.classList.add('expanded');
                        btn.classList.add('rotated');
                        icon.className = 'fas fa-chevron-up';
                    }
                });

                expandAllBtn.innerHTML = allExpanded ?
                    '<i class="fas fa-expand-alt"></i> Expand All' :
                    '<i class="fas fa-compress-alt"></i> Collapse All';
            });
        }
    });

    navLinks.forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();

            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');

            const targetId = this.getAttribute('href').substring(1);
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === targetId) {
                    section.classList.add('active');
                }
            });

            history.pushState(null, '', '#' + targetId);
        });
    });

    const sortButtons = document.querySelectorAll('.sort-btn');
    sortButtons.forEach(button => {
        button.addEventListener('click', function () {
            const sortType = this.getAttribute('data-sort');
            sortComparisonCards(sortType);

            sortButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
        });
    });

    document.querySelectorAll('.toggle-details-btn').forEach(button => {
        button.addEventListener('click', function () {
            const card = this.closest('.comparison-card');
            const details = card.querySelector('.metric-details');
            details.style.display = details.style.display === 'none' ? 'grid' : 'none';
            this.textContent = details.style.display === 'none' ? 'Show Details' : 'Hide Details';
        });
    });

    document.querySelectorAll('.details-header').forEach(header => {
        header.addEventListener('click', function () {
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

    function sortComparisonCards(sortType) {
        const container = document.querySelector('.comparison-grid');
        const cards = Array.from(container.querySelectorAll('.comparison-card'));

        cards.sort((a, b) => {
            let aValue, bValue;

            switch (sortType) {
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

        cards.forEach(card => container.appendChild(card));
    }

    document.querySelectorAll('.summary-row').forEach(row => {
        const benchmarkName = row.getAttribute('data-benchmark');

        row.addEventListener('mouseenter', function () {
            this.style.backgroundColor = '#f0f7ff';
        });

        row.addEventListener('mouseleave', function () {
            this.style.backgroundColor = '';
        });
    });
});