# GitHub Pages Deployment Guide for NCBench Project Page

## Project Information
- **Repository**: https://github.com/heqin-zhu/NCBench
- **Current Branch**: `page`
- **Project Page URL**: https://heqin-zhu.github.io/NCBench
- **Academic Homepage**: https://heqin-zhu.github.io

## Deployment Steps

### 1. Verify Files
Ensure all necessary files are in the `pages/` directory:
- ✅ `index.html` - Main project page
- ✅ `static/` - All assets (CSS, JS, images, PDFs)
- ✅ `.nojekyll` - Tells GitHub to bypass Jekyll processing
- ✅ `plan_page.md` - Project plan documentation

### 2. Commit Changes
From the repository root:
```bash
cd /root/gitrepo/NCBench
git checkout page
git add pages/
git add CLAUDE.md
git commit -m "Add NCBench project page for GitHub Pages deployment"
```

### 3. Push to GitHub
```bash
git push origin page
```

### 4. Configure GitHub Pages

#### Option A: Using GitHub Website (Recommended)
1. Go to: https://github.com/heqin-zhu/NCBench/settings/pages
2. Under "Source", select:
   - **Source**: Deploy from a branch
   - **Branch**: `page`
   - **Folder**: `/ (root)`
3. Click "Save"
4. Your site will be available at: https://heqin-zhu.github.io/NCBench

#### Option B: Using GitHub CLI (gh)
```bash
gh repo view --web --settings/pages
# Or configure via command:
gh api repos/heqin-zhu/NCBench/pages -X POST -f source[branch]=page -f source[path]=/
```

### 5. Verify Deployment
After a few minutes, visit:
- **Main page**: https://heqin-zhu.github.io/NCBench
- **PDF**: https://heqin-zhu.github.io/NCBench/static/pdfs/NCBench.pdf

## Troubleshooting

### If the page doesn't load:
1. Check GitHub Actions deployment status:
   - Go to: https://github.com/heqin-zhu/NCBench/actions
   - Look for "pages-build-deployment" workflow

2. Verify branch name is exactly `page` (lowercase)

3. Check that `index.html` is in the root of the `pages/` directory

4. Ensure `.nojekyll` file exists in `pages/` directory

### If PDFs don't load:
1. Verify PDFs are in `static/pdfs/` directory
2. Check file permissions (should be readable)

### If images don't load:
1. Verify images are in `static/images/` directory
2. Check image filenames match exactly (case-sensitive)

## File Structure
```
NCBench/
├── pages/
│   ├── .nojekyll          # Important: Disables Jekyll processing
│   ├── index.html         # Main project page
│   ├── README.md          # Original template README
│   ├── plan_page.md       # Project modification plan
│   └── static/
│       ├── css/           # Stylesheets
│       ├── js/            # JavaScript files
│       ├── images/        # Project images (PNG, etc.)
│       ├── pdfs/          # PDF files (NCBench.pdf, poster)
│       └── videos/        # (Empty - for future videos)
```

## Custom Domain (Optional)
If you want to use a custom domain:
1. Add a `CNAME` file in `pages/` directory with your domain
2. Configure DNS records with your domain provider
3. Enable "Enforce HTTPS" in GitHub Pages settings

## Maintenance
- **To update content**: Edit files in `pages/` directory, commit, and push
- **To add new images**: Place in `static/images/`
- **To add PDFs**: Place in `static/pdfs/`
- **To update links**: Edit `index.html`

## Additional Resources
- GitHub Pages Documentation: https://docs.github.com/pages
- Bulma CSS Framework: https://bulma.io/documentation/

## Contact
For issues or questions, contact: Zhu Li (https://heqin-zhu.github.io)

---
**Last Updated**: 2024-03-19
**Status**: Ready for deployment
