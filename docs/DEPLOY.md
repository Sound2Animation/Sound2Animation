# GitHub Pages Deployment Guide

This guide will help you deploy the Sound2Motion website to GitHub Pages.

## Quick Start

1. **Enable GitHub Pages**
   - Go to your repository on GitHub
   - Navigate to Settings → Pages
   - Under "Source", select "Deploy from a branch"
   - Select branch: `main` (or your default branch)
   - Select folder: `/docs`
   - Click "Save"

2. **Wait for deployment**
   - GitHub will automatically build and deploy your site
   - Your site will be available at: `https://bananasjim.github.io/sound2motion/`
   - Initial deployment may take a few minutes

3. **Add your content**
   - Add video files to `docs/assets/videos/`
   - Add audio files to `docs/assets/audio/`
   - Update `docs/index.html` with your content
   - Commit and push - GitHub Pages will auto-update

## Adding Your Results

### Video Files

1. **Export your simulation results:**
   ```bash
   # Generate videos with audio
   python render_video.py --object 3_CeramicKoiBowl --duration 5 --eevee
   ```

2. **Copy videos to docs folder:**
   ```bash
   cp output/3_CeramicKoiBowl_final.mp4 docs/assets/videos/
   ```

3. **Update index.html:**
   Edit the video gallery section in `docs/index.html`:
   ```html
   <div class="video-card">
       <div class="video-container">
           <video controls preload="metadata">
               <source src="assets/videos/3_CeramicKoiBowl_final.mp4" type="video/mp4">
           </video>
       </div>
       <div class="video-info">
           <h3>Ceramic Koi Bowl</h3>
           <p>Material: Ceramic | Drop height: 0.8m | Duration: 5s</p>
       </div>
   </div>
   ```

### Audio Files

1. **Extract audio samples:**
   ```bash
   cp output/3_CeramicKoiBowl_audio.wav docs/assets/audio/ceramic_example.wav
   ```

2. **Update index.html:**
   ```html
   <div class="audio-card">
       <h4>Ceramic Impact</h4>
       <audio controls>
           <source src="assets/audio/ceramic_example.wav" type="audio/wav">
       </audio>
   </div>
   ```

### Images/Screenshots

Add any screenshots or images to `docs/assets/images/` and reference them in HTML:
```html
<img src="assets/images/screenshot.png" alt="Description">
```

### Paper and Academic Materials

1. **Add your paper PDF:**
   ```bash
   cp your-paper.pdf docs/assets/paper.pdf
   ```

2. **Add your poster PDF:**
   ```bash
   cp your-poster.pdf docs/assets/poster.pdf
   ```

3. **Create paper preview image:**
   - Export the first page of your paper as PNG (recommended size: 800-1000px width)
   - Save as `docs/assets/images/paper-preview.png`
   - Or use an online PDF to image converter

4. **Update citation in index.html:**
   Edit the BibTeX and citation text in the Paper section (around line 97-112):
   ```html
   @inproceedings{yourpaper2024,
     author = {Your Name and Co-Authors},
     title = {{Sound2Motion}: Your Paper Title},
     booktitle = {Conference Name (e.g., CVPR, ICCV, etc.)},
     month = {June},
     year = {2024}
   }
   ```

## Customization

### Update Repository Links

Repository links in `docs/index.html` have been configured:
- Repository: `https://github.com/BANANASJIM/sound2motion`
- Website: `https://bananasjim.github.io/sound2motion/`

### Modify Content

Edit `docs/index.html` to customize:
- **Hero section** (lines 25-36): Update title, subtitle, buttons
- **About section** (lines 40-73): Modify project description
- **Results section** (lines 77-143): Add/remove video and audio cards
- **Technical section** (lines 147-239): Update technical details
- **Footer** (lines 244-253): Add team info, links

### Styling Changes

Edit `docs/css/style.css` to modify:
- **Colors**: Change CSS variables in `:root` (lines 5-15)
- **Fonts**: Update `font-family` in `body` (line 27)
- **Layout**: Adjust grid columns, spacing, padding throughout

### Add Custom Features

Edit `docs/js/main.js` to add:
- Analytics tracking
- Video autoplay on scroll
- Custom animations
- Interactive comparisons

## Batch Adding Results

### Script to Copy All Videos

Create a helper script `copy_results.sh`:
```bash
#!/bin/bash

# Copy all final videos
for video in output/*_final.mp4; do
    if [ -f "$video" ]; then
        echo "Copying $video"
        cp "$video" docs/assets/videos/
    fi
done

# Copy all audio files
for audio in output/*_audio.wav; do
    if [ -f "$audio" ]; then
        echo "Copying $audio"
        cp "$audio" docs/assets/audio/
    fi
done

echo "Done! Now update docs/index.html with the new files."
```

Make executable and run:
```bash
chmod +x copy_results.sh
./copy_results.sh
```

### Auto-generate Video Gallery HTML

Create a Python script `generate_gallery.py`:
```python
import os
from pathlib import Path

video_dir = Path('docs/assets/videos')
videos = sorted(video_dir.glob('*.mp4'))

for video in videos:
    name = video.stem.replace('_final', '').replace('_', ' ').title()
    print(f'''
<div class="video-card">
    <div class="video-container">
        <video controls preload="metadata">
            <source src="assets/videos/{video.name}" type="video/mp4">
        </video>
    </div>
    <div class="video-info">
        <h3>{name}</h3>
        <p>Material: TBD | Drop height: TBD | Duration: 5s</p>
    </div>
</div>
''')
```

Run it and copy output to `docs/index.html`:
```bash
python generate_gallery.py >> video_cards.html
```

## File Size Considerations

GitHub Pages has limits:
- **Repository size**: < 1 GB recommended
- **File size**: < 100 MB per file
- **Site size**: < 1 GB total

### Compress Videos

If your videos are too large, compress them:
```bash
# Using ffmpeg
ffmpeg -i input.mp4 -c:v libx264 -crf 28 -preset slow -c:a aac -b:a 128k output.mp4

# Or batch compress
for video in docs/assets/videos/*.mp4; do
    ffmpeg -i "$video" -c:v libx264 -crf 28 -preset slow -c:a aac -b:a 128k "${video%.mp4}_compressed.mp4"
done
```

### Use External Hosting

For very large files, consider:
- **YouTube**: Upload videos and embed with `<iframe>`
- **Google Drive**: Share files and embed
- **Cloudinary**: Free tier for media hosting

Example YouTube embed:
```html
<div class="video-container">
    <iframe
        src="https://www.youtube.com/embed/VIDEO_ID"
        frameborder="0"
        allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen>
    </iframe>
</div>
```

## Deployment Checklist

Before deploying:
- [ ] Update all placeholder links with actual repository URL
- [ ] Add your video files to `docs/assets/videos/`
- [ ] Add your audio files to `docs/assets/audio/`
- [ ] Update video gallery in `index.html` with actual files
- [ ] Test locally by opening `docs/index.html` in browser
- [ ] Commit and push all changes
- [ ] Enable GitHub Pages in repository settings
- [ ] Verify deployment at your GitHub Pages URL

## Local Testing

Test the site locally before deploying:

### Option 1: Python HTTP Server
```bash
cd docs
python -m http.server 8000
# Visit http://localhost:8000
```

### Option 2: Node.js http-server
```bash
npm install -g http-server
cd docs
http-server
# Visit http://localhost:8080
```

### Option 3: VSCode Live Server
- Install "Live Server" extension
- Right-click `docs/index.html`
- Select "Open with Live Server"

## Updating the Site

After initial deployment, updates are automatic:
```bash
# Make your changes
git add docs/
git commit -m "Update website content"
git push

# GitHub Pages will auto-deploy (takes 1-2 minutes)
```

## Custom Domain (Optional)

To use a custom domain like `sound2motion.com`:

1. **Add CNAME file:**
   ```bash
   echo "yourdomain.com" > docs/CNAME
   git add docs/CNAME
   git commit -m "Add custom domain"
   git push
   ```

2. **Configure DNS:**
   Add these DNS records at your domain registrar:
   ```
   Type: CNAME
   Name: www
   Value: bananasjim.github.io
   ```

3. **Update GitHub settings:**
   - Go to Settings → Pages
   - Enter your custom domain
   - Enable "Enforce HTTPS"

## Troubleshooting

### Site not loading
- Check GitHub Actions tab for build errors
- Verify `/docs` folder is selected in Pages settings
- Wait 5-10 minutes after enabling Pages

### Videos not playing
- Check file paths are correct (case-sensitive)
- Verify files are in `docs/assets/videos/`
- Check browser console for errors (F12)
- Try different video format (H.264 is most compatible)

### Styling looks broken
- Clear browser cache (Ctrl+Shift+R)
- Check `style.css` path in `index.html`
- Verify no CSS syntax errors

### Changes not appearing
- GitHub Pages caching can take a few minutes
- Force refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
- Check git push was successful: `git log --oneline -1`

## Advanced Features

### Analytics

Add Google Analytics to track visitors:
```html
<!-- Add to <head> in index.html -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### Social Media Cards

Add Open Graph meta tags for better sharing:
```html
<!-- Add to <head> in index.html -->
<meta property="og:title" content="Sound2Motion - Physics Simulation with Sound Synthesis">
<meta property="og:description" content="GPU-accelerated rigid body simulation with impulse-driven sound synthesis">
<meta property="og:image" content="https://bananasjim.github.io/sound2motion/assets/images/preview.png">
<meta property="og:url" content="https://bananasjim.github.io/sound2motion/">
<meta name="twitter:card" content="summary_large_image">
```

### Comments Section

Add GitHub Discussions integration:
```html
<!-- Add before closing </body> -->
<script src="https://giscus.app/client.js"
        data-repo="BANANASJIM/sound2motion"
        data-repo-id="YOUR_REPO_ID"
        data-category="General"
        data-category-id="YOUR_CATEGORY_ID"
        data-mapping="pathname"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="bottom"
        data-theme="light"
        data-lang="en"
        crossorigin="anonymous"
        async>
</script>
```

## Support

For issues or questions:
- GitHub Pages Docs: https://docs.github.com/en/pages
- Repository Issues: Create an issue in your repo
- Community: Stack Overflow with tag `github-pages`
