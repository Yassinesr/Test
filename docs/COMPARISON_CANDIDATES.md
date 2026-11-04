# Comparison candidates (surveyed 2026-11-04)

Target: beat every retrained comparison model on all five test splits, with at
least one win above 1.0 mDice. All rows must be **retrained on this
repository's manifest** (re-annotated 1288/162 split) before comparison; the
published numbers below are context only and are not comparable to any run
here. Published values are from the papers' own tables under the PraNet
protocol unless noted. "not found" means not confirmed from a primary source.

| method | venue (CCF) | backbone | Kvasir | ClinicDB | ColonDB | CVC-300 | ETIS | training code |
|---|---|---|---|---|---|---|---|---|
| Polyp-PVT | CAAI AIR 2023 (baseline) | PVTv2-b2 | .917 | .937 | .808 | .900 | .787 | this repo, arm A0 |
| PVT-CASCADE | WACV 2023 (not ranked) | PVTv2-b2 | .911 | not found | .789 | not found | .786 | github.com/SLDGroup/CASCADE |
| SEPNet | IEEE TCSVT 2024 (B) | PVTv2 | not found | not found | not found | not found | not found | github.com/wangtong627/SEPNet |
| EMCAD (PVT-EMCAD-B2) | CVPR 2024 (A) | PVTv2-b2 | +1.19 vs Polyp-PVT | +1.08 | +0.78 | not found | +2.36 | github.com/SLDGroup/EMCAD (polyp code Jan 2026) |
| CAFE-Net | ESWA 2024 (C) | PVTv2 | .918 | .939 | .798 | not found | .797 | repo unreachable |
| BMANet | 2025, venue unverified | not found | .925 | not found | .804 | not found | .824 | not verified |
| CD-PolypNet | MICCAI 2025 (B) | SAM ViT-L, 1024 px | .933 | .945 | .889 | .913 | .903 | github.com/ChangpengYue/CD-PolypNet |

## Recommendation

* **EMCAD** is the CCF-A target: same PVTv2-b2 backbone and 352 recipe as A0,
  public training code, claims one to two points over Polyp-PVT. Its paper
  reports polyp results partly on an in-distribution split and only as deltas;
  take absolute numbers from a retrain, never from the paper.
* **SEPNet** is the CCF-B target: same backbone family, clean repo, released
  weights.
* **CD-PolypNet** is not a fair bar: SAM ViT-L at 1024x1024, about twelve
  points above the PVTv2 field on ETIS. Cite it with the backbone and
  resolution difference captioned; do not treat it as the comparison.
* ESPNet (MICCAI 2025) uses a different benchmark set and is not a direct fit.

## Sources

* GFANet comparison tables: https://pmc.ncbi.nlm.nih.gov/articles/PMC13481947/
* CD-PolypNet: https://papers.miccai.org/miccai-2025/paper/2291_paper.pdf
* ESPNet: https://papers.miccai.org/miccai-2025/paper/4428_paper.pdf
* EMCAD: https://arxiv.org/abs/2405.06880
