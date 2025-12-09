# Sound2Motion Website

This directory contains the GitHub Pages website for Sound2Motion.

## Structure

```
docs/
├── index.html          # Main website page
├── css/
│   └── style.css       # Styling
├── js/
│   └── main.js         # Interactive features
├── assets/
│   ├── videos/         # Add your simulation videos here (.mp4)
│   ├── audio/          # Add your audio samples here (.wav)
│   └── images/         # Add screenshots/images here
├── DEPLOY.md           # Full deployment guide
└── README.md           # This file
```

## Quick Start

1. **Add your content:**
   - Copy video files to `assets/videos/`
   - Copy audio files to `assets/audio/`
   - Update `index.html` with your content

2. **Enable GitHub Pages:**
   - Go to repo Settings → Pages
   - Source: Deploy from branch `main`
   - Folder: `/docs`
   - Save

3. **View your site:**
   - URL: `https://bananasjim.github.io/sound2motion/`
   - Updates deploy automatically on push

## Local Testing

```bash
cd docs
python -m http.server 8000
# Visit http://localhost:8000
```

## Documentation

See [DEPLOY.md](DEPLOY.md) for complete deployment instructions, customization options, and troubleshooting.

## Features

- Responsive design (mobile, tablet, desktop)
- Paper and citation section with BibTeX
- Video gallery with lazy loading
- Audio sample players
- Smooth scrolling navigation
- Material properties table
- Technical details section
- Modern, clean UI

## Customization

- **Colors**: Edit CSS variables in `css/style.css`
- **Content**: Edit sections in `index.html`
- **Features**: Add functionality in `js/main.js`

## Adding Results

### Single object:
```bash
python render_video.py --object 3_CeramicKoiBowl --duration 5 --eevee
cp output/3_CeramicKoiBowl_final.mp4 docs/assets/videos/
```

### Batch:
```bash
python batch_render.py --eevee
cp output/*_final.mp4 docs/assets/videos/
cp output/*_audio.wav docs/assets/audio/
```

### Paper materials:
```bash
# Add paper PDF
cp your-paper.pdf docs/assets/paper.pdf

# Add poster PDF
cp your-poster.pdf docs/assets/poster.pdf

# Add paper preview image (first page as PNG)
cp paper-preview.png docs/assets/images/paper-preview.png
```

Then update `index.html` with the new files and citation information!
