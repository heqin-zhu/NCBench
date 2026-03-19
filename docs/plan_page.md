# NCBench Project Page Modification Plan

## Overview
This plan outlines the modifications needed to transform the generic academic project page template into a dedicated page for the NCBench dataset and NCfold method for RNA non-canonical base pair prediction.

## Resources Available
- **Paper**: NCBench.pdf (comprehensive paper on NCBench dataset and NCfold method)
- **Images in static/images/**:
  - `overview-NCBench-NCfold.png` - Main overview figure
  - `Comparison-canonical-methods.png` - Performance comparison
  - `IsoScore-RFM.png` - IsoScore analysis
  - `top-k.png` - Top-k selection results
  - `visualization.png` - Data visualization
  - `visualization2.png` - Additional visualization
  - `poster_NCBench_00.png` - Research poster

## Paper Information (from NCBench.pdf)
- **Title**: NCBench: A Comprehensive Benchmark Dataset and Deep Learning Approach for RNA Non-Canonical Base Pair Prediction
- **Authors**: Zhu Li, Jie Tang, Lyuwei Wang, Zhen Wu, Yuyu Xing, Xiaowei Lin, Qianqian Yuan, Jun Yu
- **Affiliations**: Ministry of Education Key Laboratory of Computational Biology, CAS-MPG Partner Institute for Computational Biology, Shanghai Institute of Nutrition and Health, Chinese Academy of Sciences, University of Chinese Academy of Sciences, etc.
- **Venue**: bioRxiv preprint
- **DOI**: 10.1101/2024.10.20.619325
- **Year**: 2024

## Modification Plan

### 1. Meta Tags Section (Lines 7-64)
**Changes needed:**
- Replace `PAPER_TITLE` with "NCBench: A Comprehensive Benchmark Dataset and Deep Learning Approach for RNA Non-Canonical Base Pair Prediction"
- Replace `AUTHOR_NAMES` with "Zhu Li, Jie Tang, Lyuwei Wang, et al."
- Replace `BRIEF_DESCRIPTION` with: "NCBench introduces a benchmark dataset with 925 RNA sequences containing 6,708 high-quality non-canonical base pair annotations, along with NCfold, a deep learning framework for accurate NC base pair prediction."
- Replace `KEYWORDS` with: "RNA structure prediction, Non-canonical base pairs, Deep learning, RNA foundation models, Bioinformatics, Computational biology"
- Replace `INSTITUTION_OR_LAB_NAME` with "CAS-MPG Partner Institute for Computational Biology"
- Update `citation_author` fields with actual author names
- Update `citation_publication_date` to "2024"
- Update `citation_conference_title` to "bioRxiv"

### 2. Structured Data (Lines 112-179)
**Changes needed:**
- Update author list with all 8 authors
- Update affiliations
- Update abstract with paper's actual abstract
- Update keywords
- Update BibTeX citation
- Update `about` section with relevant research areas

### 3. Hero Section (Lines 238-314)
**Changes needed:**
- Update `publication-title` to "NCBench: A Comprehensive Benchmark Dataset and Deep Learning Approach for RNA Non-Canonical Base Pair Prediction"
- Update `publication-authors` with all authors:
  - Zhu Li<sup>*</sup>
  - Jie Tang<sup>*</sup>
  - Lyuwei Wang
  - Zhen Wu
  - Yuyu Xing
  - Xiaowei Lin
  - Qianqian Yuan
  - Jun Yu
- Update institution to "Shanghai Institute of Nutrition and Health, Chinese Academy of Sciences"
- Update venue to "bioRxiv 2024"
- Add equal contribution note for first two authors

**Links to update:**
- Paper link: Use `static/pdfs/NCBench.pdf` or bioRxiv link
- Code link: Add GitHub repository link when available
- arXiv link: Add when available

### 4. Teaser Section (Lines 318-333)
**Changes needed:**
- Replace video with static image using `overview-NCBench-NCfold.png`
- Update subtitle to describe the NCfold framework

### 5. Abstract Section (Lines 336-351)
**Changes needed:**
- Replace placeholder with paper's abstract:
  > Non-canonical (NC) base pairs play crucial roles in RNA structural stability, functionality, and recognition processes. Despite their importance, limited benchmark resources and deep learning approaches have hindered accurate NC base pair prediction. We introduce NCBench, a comprehensive benchmark dataset containing 925 RNA sequences with 6,708 high-quality NC annotations curated from protein-RNA complexes and RNA-only 3D structures. NCBench features rigorous annotation standards, redundancy reduction, and multiple RNA types. Additionally, we present NCfold, a novel deep learning framework that fuses sequence features with structural priors from RNA foundation models. NCfold employs a dual-branch architecture with Representative Embedding Fusion (REF) to integrate multiple RNA foundation models and uses Base Pair Motif energy matrices as structural priors. Extensive experiments demonstrate NCfold achieves state-of-the-art performance, with the AttnMatFusion_net variant significantly outperforming existing methods. Analysis reveals that longer training sequences and careful foundation model selection improve performance, particularly for G-U wobble pairs and multi-branch loops. NCBench and NCfold provide valuable resources and methodologies for RNA structural biology research.

### 6. Image Carousel Section (Lines 355-392)
**Changes needed:**
Replace carousel items with paper figures:
- Item 1: `overview-NCBench-NCfold.png` - "Overview of the NCfold framework"
- Item 2: `Comparison-canonical-methods.png` - "Performance comparison with existing methods"
- Item 3: `IsoScore-RFM.png` - "IsoScore distribution across RNA foundation models"
- Item 4: `top-k.png` - "Top-k foundation model selection strategy"

### 7. YouTube Video Section (Lines 398-415)
**Decision:** Keep but update with placeholder or remove if no video available
- For now: Add note "Video presentation coming soon" or remove section

### 8. Video Carousel Section (Lines 419-449)
**Decision:** Remove this section unless videos are available

### 9. Poster Section (Lines 456-469)
**Changes needed:**
- Update poster PDF reference to `poster_NCBench_00.png`

### 10. BibTeX Citation Section (Lines 473-492)
**Changes needed:**
Update BibTeX entry:
```bibtex
@article{li2024ncbench,
  title={NCBench: A Comprehensive Benchmark Dataset and Deep Learning Approach for RNA Non-Canonical Base Pair Prediction},
  author={Li, Zhu and Tang, Jie and Wang, Lyuwei and Wu, Zhen and Xing, Yuyu and Lin, Xiaowei and Yuan, Qianqian and Yu, Jun},
  journal={bioRxiv},
  year={2024},
  publisher={Cold Spring Harbor Laboratory},
  doi={10.1101/2024.10.20.619325},
  url={https://www.biorxiv.org/content/10.1101/2024.10.20.619325}
}
```

### 11. Footer Section (Lines 495-511)
**Changes needed:**
- Keep template attribution
- Update license information if needed

## Additional Content to Add

### Method Section
Consider adding sections for:
- Dataset description (925 sequences, 6,708 NC pairs)
- NCfold architecture details (dual-branch, REF, BPM energy)
- Foundation models used (structRFM, rnaernie, rnafm, etc.)

### Results Section
Consider adding:
- Performance metrics (F1, MCC)
- Comparison tables
- Visualization results

## File Operations Summary
1. Read `/root/gitrepo/NCBench/pages/index.html` ✓
2. Create `/root/gitrepo/NCBench/pages/plan_page.md` ✓
3. Modify `/root/gitrepo/NCBench/pages/index.html` with all planned changes ✓
4. Add CSS styling for centered images in teaser section ✓

## Completed Modifications

### ✓ Meta Tags Section
- Updated paper title, authors, and description
- Added relevant keywords for RNA structure prediction
- Updated Open Graph and Twitter card meta tags
- Added academic citation metadata with all 8 authors
- Added DOI reference

### ✓ Structured Data (JSON-LD)
- Updated scholarly article schema with complete author list
- Added proper abstract from the paper
- Updated keywords and research areas
- Added organization data for CAS-MPG Partner Institute

### ✓ Hero Section
- Updated publication title with full NCBench title
- Listed all 8 authors with links
- Added equal contribution notation
- Updated institution and venue information
- Added functional links: Paper (PDF), bioRxiv, Code (GitHub), DOI

### ✓ Teaser Section
- Replaced video with static overview image
- Added descriptive caption for NCfold framework

### ✓ Abstract Section
- Added complete abstract from NCBench paper
- Describes dataset, NCfold method, and key findings

### ✓ Image Carousel
- Updated with 4 relevant figures from the paper:
  1. NCfold Framework Overview
  2. Performance Comparison
  3. IsoScore Distribution
  4. Top-k Selection Strategy

### ✓ Video Section
- Updated to show "Video presentation coming soon" placeholder

### ✓ Poster Section
- Updated to display NCBench poster image

### ✓ BibTeX Citation
- Updated with correct citation for NCBench paper
- Includes DOI and bioRxiv URL

### ✓ More Works Section
- Simplified to show lab information link

### ✓ CSS Enhancement
- Added `.center-image` class for proper image alignment in teaser section

## Notes
- All text content is in English as requested
- Images are already in place in `static/images/`
- PDF is already in place at `NCBench.pdf` (root of pages directory)
- Responsive design and accessibility features maintained
- All functional elements (copy button, scroll to top, carousel) preserved
- Template attribution maintained in footer

## Next Steps (Optional)
- Update placeholder URLs (YOUR_DOMAIN.com) with actual domain when deployed
- Add actual author profile links when available
- Add video presentation when ready
- Consider adding a method section with more detailed technical information
- Add results/comparison tables if desired
