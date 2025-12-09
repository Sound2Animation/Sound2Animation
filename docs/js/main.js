/**
 * Sound2Motion Website - Main JavaScript
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    initSmoothScrolling();
    initVideoLazyLoading();
    initScrollAnimations();
    initMobileMenu();
    initVideoPlayPause();
    initPaperImageFallback();
});

/**
 * Smooth scrolling for navigation links
 */
function initSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');

    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');

            // Skip if it's just "#"
            if (href === '#') return;

            const target = document.querySelector(href);
            if (target) {
                e.preventDefault();
                const offsetTop = target.offsetTop - 80; // Account for fixed navbar

                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

/**
 * Lazy loading for videos to improve page load performance
 */
function initVideoLazyLoading() {
    const videos = document.querySelectorAll('video');

    // Options for intersection observer
    const options = {
        root: null,
        rootMargin: '50px',
        threshold: 0.1
    };

    // Callback when video enters viewport
    const handleIntersection = (entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const video = entry.target;
                const source = video.querySelector('source');

                // Load the video if not already loaded
                if (source && !video.hasAttribute('data-loaded')) {
                    video.load();
                    video.setAttribute('data-loaded', 'true');
                }

                // Stop observing this video
                observer.unobserve(video);
            }
        });
    };

    // Create intersection observer
    const observer = new IntersectionObserver(handleIntersection, options);

    // Observe all videos
    videos.forEach(video => {
        observer.observe(video);
    });
}

/**
 * Scroll animations for sections
 */
function initScrollAnimations() {
    const elements = document.querySelectorAll('.feature-card, .video-card, .technical-card, .audio-card');

    const options = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const handleScroll = (entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    };

    const observer = new IntersectionObserver(handleScroll, options);

    elements.forEach(element => {
        observer.observe(element);
    });
}

/**
 * Mobile menu toggle (if needed for responsive design)
 */
function initMobileMenu() {
    // Check if we need a hamburger menu for mobile
    // This is a placeholder - you can add hamburger menu functionality if needed

    const handleResize = () => {
        const width = window.innerWidth;
        const navMenu = document.querySelector('.nav-menu');

        if (width > 768) {
            navMenu.style.display = 'flex';
        }
    };

    window.addEventListener('resize', handleResize);
    handleResize();
}

/**
 * Video play/pause control enhancement
 * Pause other videos when one starts playing
 */
function initVideoPlayPause() {
    const videos = document.querySelectorAll('video');

    videos.forEach(video => {
        video.addEventListener('play', function() {
            // Pause all other videos
            videos.forEach(otherVideo => {
                if (otherVideo !== video && !otherVideo.paused) {
                    otherVideo.pause();
                }
            });
        });

        // Add custom controls styling
        video.addEventListener('loadedmetadata', function() {
            console.log(`Video loaded: ${video.querySelector('source')?.src}`);
        });

        // Handle video errors gracefully
        video.addEventListener('error', function(e) {
            const source = video.querySelector('source');
            const videoCard = video.closest('.video-card');

            if (videoCard) {
                const errorMsg = document.createElement('div');
                errorMsg.className = 'video-error';
                errorMsg.style.padding = '2rem';
                errorMsg.style.textAlign = 'center';
                errorMsg.style.color = '#ef4444';
                errorMsg.textContent = 'Video not available yet. Add your rendered videos to docs/assets/videos/';

                video.parentElement.appendChild(errorMsg);
            }
        });
    });
}

/**
 * Audio player enhancements
 */
function initAudioPlayers() {
    const audioElements = document.querySelectorAll('audio');

    audioElements.forEach(audio => {
        // Pause other audio when one starts playing
        audio.addEventListener('play', function() {
            audioElements.forEach(otherAudio => {
                if (otherAudio !== audio && !otherAudio.paused) {
                    otherAudio.pause();
                }
            });
        });

        // Handle audio errors
        audio.addEventListener('error', function() {
            console.warn('Audio file not found:', audio.querySelector('source')?.src);
        });
    });
}

/**
 * Navbar background change on scroll
 */
function initNavbarScroll() {
    const navbar = document.querySelector('.navbar');

    window.addEventListener('scroll', () => {
        if (window.scrollY > 100) {
            navbar.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
        } else {
            navbar.style.boxShadow = '0 1px 2px 0 rgba(0, 0, 0, 0.05)';
        }
    });
}

// Initialize audio players and navbar scroll
initAudioPlayers();
initNavbarScroll();

/**
 * Utility function to check if element is in viewport
 */
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

/**
 * Add "Back to Top" button functionality
 */
function initBackToTop() {
    const backToTopLinks = document.querySelectorAll('a[href="#home"]');

    backToTopLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    });
}

/**
 * Handle paper preview image fallback if image doesn't exist
 */
function initPaperImageFallback() {
    const paperImg = document.querySelector('.paper-thumbnail img');

    if (paperImg) {
        paperImg.addEventListener('error', function() {
            // Create placeholder div
            const placeholder = document.createElement('div');
            placeholder.className = 'paper-placeholder';
            placeholder.innerHTML = `
                <div style="
                    width: 100%;
                    height: 400px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    padding: 2rem;
                    text-align: center;
                    border: 1px solid #e5e7eb;
                ">
                    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="16" y1="13" x2="8" y2="13"></line>
                        <line x1="16" y1="17" x2="8" y2="17"></line>
                        <polyline points="10 9 9 9 8 9"></polyline>
                    </svg>
                    <h3 style="margin-top: 1rem; font-size: 1.2rem;">Paper Preview</h3>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.9;">
                        Add your paper preview image:<br>
                        <code style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.5rem; border-radius: 3px;">
                            docs/assets/images/paper-preview.png
                        </code>
                    </p>
                </div>
            `;

            // Replace image with placeholder
            this.parentElement.innerHTML = '';
            this.parentElement.appendChild(placeholder);
        });
    }
}

initBackToTop();
