# not-color-blind

Code repository for [Not Color Blind: AI Predicts Racial Identity from Black and White Retinal Vessel Segmentations](). Authored by Aaron S. Coyner, PhD, Praveer Singh, PhD, James M. Brown, PhD, Susan Ostmo MS, R.V.Paul Chan MD, Michael F. Chiang MD, MA, Jayashree Kalpathy-Cramer PhD, and J. Peter Campbell MD, MPH. ASC and PS contributed to this work equally. JKC and JPC supervised this work equally.
## Abstract

### Background
Artificial intelligence (AI) may demonstrate racial biases when skin or choroidal pigmentation is present in medical images. Recent studies have shown that convolutional neural networks (CNNs) can predict race from images that were not previously thought to contain race-specific features. We evaluate whether grayscale retinal vessel maps (RVMs) of patients screened for retinopathy of prematurity (ROP) contain race-specific features.
					
### Methods
4095 retinal fundus images (RFIs) were collected from 245 Black and White infants. A U-Net generated RVMs from RFIs, which were subsequently thresholded, binarized, or skeletonized. To determine whether RVM differences between Black and White eyes were physiological, CNNs were trained to predict race from color RFIs, raw RVMs, and thresholded, binarized, or skeletonized RVMs. Area under the precision-recall curve (AUC-PR) was evaluated.
	
### Findings
CNNs predicted race from RFIs near perfectly (image-level AUC-PR: 0.999, subject-level AUC-PR: 1.000). Raw RVMs were almost as informative as color RFIs (image-level AUC-PR: 0.938, subject-level AUC-PR: 0.995). Ultimately, CNNs were able to detect whether RFIs or RVMs were from Black or White babies, regardless of whether images contained color, vessel segmentation brightness differences were nullified, or vessel segmentation widths were normalized.
					
### Interpretation
AI can detect race from grayscale RVMs that were not thought to contain racial information. Two potential explanations for these findings are that: retinal vessels physiologically differ between Black and White babies or the U-Net segments the retinal vasculature differently for various fundus pigmentations. Either way, the implications remain the same: AI algorithms have potential to demonstrate racial bias in practice, even when preliminary attempts to remove such information from the underlying images appear to be successful.
