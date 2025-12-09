# Enable GitHub Pages - Step by Step

## Current Status
✅ Files are correctly placed in `docs/` folder
✅ Files are committed and pushed to GitHub
❌ **GitHub Pages needs to be enabled** ← YOU ARE HERE

## Step-by-Step Instructions

### Step 1: Go to Repository Settings

Click this link (replace with your actual repository):
```
https://github.com/BANANASJIM/sound2motion/settings/pages
```

Or navigate manually:
1. Go to your repository: https://github.com/BANANASJIM/sound2motion
2. Click "Settings" tab (top right)
3. Click "Pages" in the left sidebar

### Step 2: Configure GitHub Pages

You should see a page that says "GitHub Pages"

Look for the section "Build and deployment":

```
┌─────────────────────────────────────────────────────┐
│ Build and deployment                                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│ Source                                               │
│ ┌─────────────────────────────────────────────┐    │
│ │ Deploy from a branch               ▼        │    │
│ └─────────────────────────────────────────────┘    │
│                                                      │
│ Branch                                               │
│ ┌──────────┐ ┌───────────┐  [Save]                │
│ │ main  ▼  │ │ /docs  ▼  │                         │
│ └──────────┘ └───────────┘                         │
│                                                      │
└─────────────────────────────────────────────────────┘
```

**IMPORTANT SETTINGS:**
1. **Source**: Select "Deploy from a branch"
2. **Branch**: Select "main" (or your default branch)
3. **Folder**: Select "/docs" ← **NOT "/ (root)"**
4. Click the **"Save"** button

### Step 3: Wait for Deployment

After clicking Save, you'll see:
```
✓ Your site is published at https://bananasjim.github.io/sound2motion/
```

**Deployment takes 2-5 minutes**. You can check progress:
1. Go to: https://github.com/BANANASJIM/sound2motion/actions
2. Look for "pages build and deployment" workflow
3. Wait for the green checkmark ✓

### Step 4: Visit Your Website

Once deployed, visit:
```
https://bananasjim.github.io/sound2motion/
```

If you still see 404:
- Wait 1-2 more minutes
- Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
- Try incognito/private browsing mode

## Troubleshooting

### "I don't see the Pages option in Settings"

**Cause**: Repository might be private

**Solution**:
1. Go to https://github.com/BANANASJIM/sound2motion/settings
2. Scroll down to "Danger Zone"
3. Click "Change repository visibility"
4. Select "Public"
5. Confirm the change
6. Go back to Settings → Pages

### "I selected the settings but still get 404"

**Cause**: Deployment hasn't finished yet

**Solution**:
1. Check https://github.com/BANANASJIM/sound2motion/actions
2. Wait for the workflow to complete (green checkmark)
3. Usually takes 2-5 minutes
4. Hard refresh your browser after deployment completes

### "The branch dropdown doesn't show 'main'"

**Cause**: Different default branch name

**Solution**:
- Check your branch name at https://github.com/BANANASJIM/sound2motion
- Select whatever branch contains your code (might be "master" instead of "main")
- Make sure the `/docs` folder exists on that branch

## Verification

After enabling GitHub Pages, verify:

1. **Settings page shows**:
   ```
   ✓ Your site is live at https://bananasjim.github.io/sound2motion/
   ```

2. **Actions page shows**:
   - "pages build and deployment" workflow
   - Green checkmark ✓

3. **Website loads**:
   - Navigation bar visible
   - "Sound2Motion" title
   - Sections: Home, About, Paper, Results, Technical

## Common Mistakes

❌ **Wrong folder selected**:
   - CORRECT: `/docs`
   - WRONG: `/ (root)`

❌ **Wrong branch selected**:
   - CORRECT: `main` (or whatever branch has your code)
   - WRONG: `gh-pages` (unless you specifically created this branch)

❌ **Not clicking "Save"**:
   - Must click the "Save" button after selecting options

❌ **Repository is private**:
   - GitHub Pages requires public repositories (free accounts)
   - Make repository public first

## Expected Timeline

- **T+0 minutes**: Click "Save" in Pages settings
- **T+1 minute**: GitHub Actions workflow starts
- **T+2-5 minutes**: Website goes live
- **T+5 minutes**: Website fully accessible at https://bananasjim.github.io/sound2motion/

## Need Help?

If still having issues after 10 minutes:
1. Check repository is public
2. Check `docs/index.html` exists on GitHub
3. Check Actions tab for error messages
4. Try disabling and re-enabling GitHub Pages
