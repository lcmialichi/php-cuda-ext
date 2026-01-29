document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('.report-section');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
            
            const sectionId = this.getAttribute('data-section');
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === sectionId) {
                    section.classList.add('active');
                }
            });
        });
    });
    
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            const testCards = document.querySelectorAll('.test-card');
            
            testCards.forEach(card => {
                const testName = card.querySelector('h4').textContent.toLowerCase();
                if (testName.includes(searchTerm)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    }
    
    const benchmarkFilter = document.querySelector('.benchmark-filter');
    if (benchmarkFilter) {
        benchmarkFilter.addEventListener('change', function() {
            const selectedBenchmark = this.value;
            const benchmarkGroups = document.querySelectorAll('.benchmark-group');
            
            benchmarkGroups.forEach(group => {
                const benchmarkName = group.getAttribute('data-benchmark');
                if (!selectedBenchmark || benchmarkName === selectedBenchmark) {
                    group.style.display = 'block';
                } else {
                    group.style.display = 'none';
                }
            });
        });
    }
    
    const sortSelect = document.querySelector('.sort-select');
    if (sortSelect) {
        sortSelect.addEventListener('change', function() {
            const sortValue = this.value;
            const testCards = Array.from(document.querySelectorAll('.test-card'));
            
            testCards.sort((a, b) => {
                const aName = a.getAttribute('data-name');
                const bName = b.getAttribute('data-name');
                const aTime = parseFloat(a.getAttribute('data-time'));
                const bTime = parseFloat(b.getAttribute('data-time'));
                const aMemory = parseFloat(a.getAttribute('data-memory'));
                const bMemory = parseFloat(b.getAttribute('data-memory'));
                const aOps = parseFloat(a.getAttribute('data-ops'));
                const bOps = parseFloat(b.getAttribute('data-ops'));
                const aRuns = parseInt(a.getAttribute('data-runs'));
                const bRuns = parseInt(b.getAttribute('data-runs'));
                
                switch(sortValue) {
                    case 'name-asc':
                        return aName.localeCompare(bName);
                    case 'name-desc':
                        return bName.localeCompare(aName);
                    case 'time-asc':
                        return aTime - bTime;
                    case 'time-desc':
                        return bTime - aTime;
                    case 'memory-asc':
                        return aMemory - bMemory;
                    case 'memory-desc':
                        return bMemory - aMemory;
                    case 'ops-desc':
                        return bOps - aOps;
                    case 'ops-asc':
                        return aOps - bOps;
                    default:
                        return 0;
                }
            });
            
            const testsGrid = document.querySelector('.tests-grid');
            if (testsGrid) {
                testCards.forEach(card => testsGrid.appendChild(card));
            }
        });
    }
    
    const viewDetailsBtns = document.querySelectorAll('.view-details-btn');
    const detailsModal = document.getElementById('detailsModal');
    const modalBody = document.getElementById('modalBody');
    const closeModal = detailsModal?.querySelector('.close-modal');
    const modalOverlay = detailsModal?.querySelector('.modal-overlay');
    
    const benchmarkData = window.benchmarkData || {};
    
    viewDetailsBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const testName = this.getAttribute('data-test');
            const testCard = this.closest('.test-card');
            const benchmarkGroup = testCard?.closest('.benchmark-group');
            const handlerName = benchmarkGroup?.getAttribute('data-benchmark');
            
            const testData = findTestData(testName, handlerName);
            
            if (testData && modalBody) {
                const detailsHtml = generateTestDetailsHTML(testData, testName, handlerName);
                modalBody.innerHTML = detailsHtml;
                
                modalBody.querySelectorAll('.view-run-details').forEach(runBtn => {
                    runBtn.addEventListener('click', function() {
                        const runData = JSON.parse(this.getAttribute('data-run'));
                        showRunDetails(runData, testName);
                    });
                });
                
                detailsModal.style.display = 'block';
                document.body.style.overflow = 'hidden';
            }
        });
    });
    
    if (closeModal) {
        closeModal.addEventListener('click', closeDetailsModal);
    }
    
    if (modalOverlay) {
        modalOverlay.addEventListener('click', closeDetailsModal);
    }
    
    function closeDetailsModal() {
        detailsModal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
    
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && detailsModal.style.display === 'block') {
            closeDetailsModal();
        }
    });
    
    function findTestData(testName, handlerName) {
        console.log('Looking for:', testName, handlerName);
        return null;
    }
    
    function generateTestDetailsHTML(testData, testName, handlerName) {
        return '<div class="loading">Loading details...</div>';
    }
    
    function showRunDetails(runData, testName) {
        alert(`Run details for ${testName}:\n` +
              `Time: ${runData.stats.time.avg} ms\n` +
              `Memory: ${runData.stats.memory.avg}\n` +
              `Ops/Sec: ${runData.stats.ops.formatted}\n` +
              `Iterations: ${runData.iterations}`);
    }
    
    const toggleButtons = document.querySelectorAll('.toggle-performance-chart');
    toggleButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const chartId = this.getAttribute('data-chart');
            const chart = document.getElementById(chartId);
            if (chart) {
                chart.style.display = chart.style.display === 'none' ? 'block' : 'none';
                this.querySelector('i').classList.toggle('fa-chevron-down');
                this.querySelector('i').classList.toggle('fa-chevron-up');
            }
        });
    });
    
    const summaryRows = document.querySelectorAll('.summary-row');
    summaryRows.forEach(row => {
        row.addEventListener('click', function() {
            const benchmark = this.getAttribute('data-benchmark');
            const benchmarkFilter = document.querySelector('.benchmark-filter');
            if (benchmarkFilter) {
                benchmarkFilter.value = benchmark;
                benchmarkFilter.dispatchEvent(new Event('change'));
                
                document.querySelector('[data-section="comparison"]').click();
            }
        });
    });
    
    function initCharts() {
        console.log('Charts initialized');
    }
    
    initCharts();
});