// Enhanced Custom JavaScript for Jorge Roldan's Blog

document.addEventListener('DOMContentLoaded', function() {
    // Enhanced theme toggle functionality
    const themeToggle = document.getElementById('theme-toggle');
    const html = document.documentElement;
    
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
            
            // Add transition class for smooth theme change
            html.classList.add('theme-transitioning');
            
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            
            // Update theme toggle icon with animation
            updateThemeIcon(newTheme);
            
            // Remove transition class after animation
            setTimeout(() => {
                html.classList.remove('theme-transitioning');
            }, 300);
        });
    }
    
    // Initialize theme with smooth transition
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
        html.setAttribute('data-theme', savedTheme);
        updateThemeIcon(savedTheme);
    }
    
    function updateThemeIcon(theme) {
        const darkIcon = document.querySelector('.theme-toggle-dark');
        const lightIcon = document.querySelector('.theme-toggle-light');
        
        if (theme === 'dark') {
            darkIcon.style.display = 'none';
            lightIcon.style.display = 'block';
            lightIcon.style.animation = 'fadeInScale 0.3s ease';
        } else {
            darkIcon.style.display = 'block';
            lightIcon.style.display = 'none';
            darkIcon.style.animation = 'fadeInScale 0.3s ease';
        }
    }
    
    // Enhanced smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const headerHeight = document.querySelector('.header').offsetHeight;
                const targetPosition = target.offsetTop - headerHeight - 20;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Enhanced hover effects for post cards with parallax
    const entries = document.querySelectorAll('.entry');
    entries.forEach((entry, index) => {
        // Add staggered animation delay
        entry.style.animationDelay = `${index * 0.1}s`;
        
        entry.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-12px) scale(1.02)';
            this.style.boxShadow = '0 25px 50px -12px rgba(0, 0, 0, 0.25)';
        });
        
        entry.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
            this.style.boxShadow = '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)';
        });
        
        // Add subtle parallax effect on scroll
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.1;
            const entryTop = entry.offsetTop;
            const entryHeight = entry.offsetHeight;
            
            if (scrolled + window.innerHeight > entryTop && scrolled < entryTop + entryHeight) {
                entry.style.transform = `translateY(${rate * 0.5}px)`;
            }
        });
    });
    
    // Enhanced typing animation for home title
    const homeTitle = document.querySelector('.home-info-title');
    if (homeTitle) {
        const text = homeTitle.textContent;
        homeTitle.textContent = '';
        homeTitle.style.opacity = '1';
        
        let i = 0;
        const typeWriter = () => {
            if (i < text.length) {
                homeTitle.textContent += text.charAt(i);
                i++;
                setTimeout(typeWriter, 80);
            } else {
                // Add cursor blink effect after typing
                homeTitle.style.borderRight = '2px solid transparent';
            }
        };
        
        // Start typing animation after a short delay
        setTimeout(typeWriter, 800);
    }
    
    // Enhanced parallax effect for header
    const header = document.querySelector('.header');
    if (header) {
        let lastScrollY = window.pageYOffset;
        
        window.addEventListener('scroll', function() {
            const currentScrollY = window.pageYOffset;
            
            if (currentScrollY > lastScrollY && currentScrollY > 100) {
                // Scrolling down - hide header
                header.style.transform = 'translateY(-100%)';
            } else {
                // Scrolling up - show header
                header.style.transform = 'translateY(0)';
            }
            
            lastScrollY = currentScrollY;
        });
    }
    
    // Enhanced intersection observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);
    
    // Observe all entries for enhanced fade-in animation
    entries.forEach((entry, index) => {
        entry.style.opacity = '0';
        entry.style.transform = 'translateY(30px)';
        entry.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
        entry.style.transitionDelay = `${index * 0.1}s`;
        observer.observe(entry);
    });
    
    // Enhanced copy button for code blocks with better feedback
    document.querySelectorAll('pre').forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
        `;
        button.title = 'Copy code';
        
        button.addEventListener('click', async () => {
            const code = block.querySelector('code');
            if (code) {
                try {
                    await navigator.clipboard.writeText(code.textContent);
                    
                    // Enhanced success feedback
                    button.innerHTML = `
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="20,6 9,17 4,12"></polyline>
                        </svg>
                    `;
                    button.title = 'Copied!';
                    button.style.background = 'rgba(16, 185, 129, 0.2)';
                    button.style.color = '#10b981';
                    
                    // Add success animation
                    button.style.animation = 'copySuccess 0.5s ease';
                    
                    setTimeout(() => {
                        button.innerHTML = `
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                            </svg>
                        `;
                        button.title = 'Copy code';
                        button.style.background = 'rgba(255, 255, 255, 0.1)';
                        button.style.color = 'inherit';
                        button.style.animation = '';
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy code:', err);
                    
                    // Error feedback
                    button.style.background = 'rgba(239, 68, 68, 0.2)';
                    button.style.color = '#ef4444';
                    button.title = 'Failed to copy';
                    
                    setTimeout(() => {
                        button.style.background = 'rgba(255, 255, 255, 0.1)';
                        button.style.color = 'inherit';
                        button.title = 'Copy code';
                    }, 2000);
                }
            }
        });
        
        block.style.position = 'relative';
        block.appendChild(button);
    });
    
    // Enhanced search functionality
    const searchToggle = document.getElementById('search-toggle');
    if (searchToggle) {
        searchToggle.addEventListener('click', function() {
            // Add click animation
            this.style.animation = 'buttonPulse 0.3s ease';
            setTimeout(() => {
                this.style.animation = '';
            }, 300);
            
            // Implement search modal or redirect to search page
            window.location.href = '/search/';
        });
    }
    
    // Enhanced keyboard shortcuts with better feedback
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + / for theme toggle
        if ((e.ctrlKey || e.metaKey) && e.key === '/') {
            e.preventDefault();
            if (themeToggle) {
                themeToggle.style.animation = 'buttonPulse 0.3s ease';
                setTimeout(() => {
                    themeToggle.style.animation = '';
                }, 300);
                themeToggle.click();
            }
        }
        
        // Ctrl/Cmd + k for search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            if (searchToggle) {
                searchToggle.style.animation = 'buttonPulse 0.3s ease';
                setTimeout(() => {
                    searchToggle.style.animation = '';
                }, 300);
                searchToggle.click();
            }
        }
    });
    
    // Enhanced loading animation
    window.addEventListener('load', function() {
        document.body.classList.add('loaded');
        
        // Add entrance animation for body
        document.body.style.animation = 'fadeInBody 0.8s ease';
    });
    
    // Enhanced scroll-based animations
    const scrollElements = document.querySelectorAll('.scroll-animate');
    
    const elementInView = (el, dividend = 1) => {
        const elementTop = el.getBoundingClientRect().top;
        return (
            elementTop <=
            (window.innerHeight || document.documentElement.clientHeight) / dividend
        );
    };
    
    const displayScrollElement = (element) => {
        element.classList.add('scrolled');
    };
    
    const hideScrollElement = (element) => {
        element.classList.remove('scrolled');
    };
    
    const handleScrollAnimation = () => {
        scrollElements.forEach((el) => {
            if (elementInView(el, 1.25)) {
                displayScrollElement(el);
            } else {
                hideScrollElement(el);
            }
        });
    };
    
    window.addEventListener('scroll', () => {
        handleScrollAnimation();
    });
    
    // Enhanced navigation hover effects
    const navLinks = document.querySelectorAll('.nav a');
    navLinks.forEach(link => {
        link.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
        });
        
        link.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Enhanced social icons with ripple effect
    const socialIcons = document.querySelectorAll('.social-icons a');
    socialIcons.forEach(icon => {
        icon.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
    
    // Enhanced reading progress with smooth animation
    const progressBar = document.getElementById('reading-progress');
    if (progressBar) {
        let ticking = false;
        
        function updateProgress() {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;
            
            progressBar.style.width = scrollPercent + '%';
            ticking = false;
        }
        
        function requestTick() {
            if (!ticking) {
                requestAnimationFrame(updateProgress);
                ticking = true;
            }
        }
        
        window.addEventListener('scroll', requestTick);
    }
    
    // Enhanced mobile menu functionality
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const mobileMenu = document.querySelector('.mobile-menu');
    
    if (mobileMenuToggle && mobileMenu) {
        mobileMenuToggle.addEventListener('click', function() {
            mobileMenu.classList.toggle('active');
            this.classList.toggle('active');
        });
    }
    
    // Enhanced image lazy loading
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
});

// Enhanced CSS animations
const style = document.createElement('style');
style.textContent = `
    .copy-button {
        position: absolute;
        top: 12px;
        right: 12px;
        background: rgba(255, 255, 255, 0.1);
        border: none;
        border-radius: 6px;
        padding: 8px;
        cursor: pointer;
        color: inherit;
        transition: all 0.3s ease;
        opacity: 0;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    pre:hover .copy-button {
        opacity: 1;
    }
    
    .copy-button:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.1);
    }
    
    @keyframes copySuccess {
        0% { transform: scale(1); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    
    @keyframes buttonPulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes fadeInScale {
        0% { 
            opacity: 0;
            transform: scale(0.8);
        }
        100% { 
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes fadeInBody {
        0% { 
            opacity: 0;
            transform: translateY(20px);
        }
        100% { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .theme-transitioning * {
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    
    .animate-in {
        animation: slideInUp 0.8s ease forwards;
    }
    
    @keyframes slideInUp {
        0% {
            opacity: 0;
            transform: translateY(30px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .scroll-animate {
        opacity: 0;
        transform: translateY(30px);
        transition: all 0.8s ease;
    }
    
    .scroll-animate.scrolled {
        opacity: 1;
        transform: translateY(0);
    }
    
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: scale(0);
        animation: ripple-animation 0.6s linear;
        pointer-events: none;
    }
    
    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .mobile-menu-toggle {
        display: none;
        flex-direction: column;
        cursor: pointer;
        padding: 8px;
    }
    
    .mobile-menu-toggle span {
        width: 25px;
        height: 3px;
        background: currentColor;
        margin: 3px 0;
        transition: 0.3s;
    }
    
    .mobile-menu-toggle.active span:nth-child(1) {
        transform: rotate(-45deg) translate(-5px, 6px);
    }
    
    .mobile-menu-toggle.active span:nth-child(2) {
        opacity: 0;
    }
    
    .mobile-menu-toggle.active span:nth-child(3) {
        transform: rotate(45deg) translate(-5px, -6px);
    }
    
    .mobile-menu {
        display: none;
        position: fixed;
        top: 60px;
        left: 0;
        right: 0;
        background: var(--theme);
        border-top: 1px solid var(--border);
        padding: 1rem;
        transform: translateY(-100%);
        transition: transform 0.3s ease;
    }
    
    .mobile-menu.active {
        transform: translateY(0);
    }
    
    .lazy {
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .lazy.loaded {
        opacity: 1;
    }
    
    @media (max-width: 768px) {
        .mobile-menu-toggle {
            display: flex;
        }
        
        .mobile-menu {
            display: block;
        }
        
        .nav-links {
            display: none;
        }
    }
    
    body.loaded {
        opacity: 1;
    }
    
    body {
        opacity: 0;
        transition: opacity 0.3s ease;
    }
`;
document.head.appendChild(style);
