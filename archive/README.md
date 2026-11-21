# SURYOKTI: Archive and Historical Bundles

This directory manages standalone compressed project packages, backup snapshots, and documented duplicate candidate files across the SURYOKTI workspace.

---

## 1. Directory Structure

```text
archive/
├── README.md                                    # Archive descriptions and usage policy (this file)
│
├── bundles/                                     # Complete portable zip snapshots
│   ├── SURYOKTI_Full_Project.zip                # Complete project zip (scripts, models, reports, datasets)
│   └── SURYOKTI_All_Datasets.zip                # Standalone consolidated zip of all 18 CSV datasets
│
└── duplicate_candidates/                        # Duplicate file inventory and hash tracking
    └── DUPLICATE_FILES_INVENTORY.md             # Hash-verified duplicate inventory and rationale
```

---

## 2. Archive Bundles Summary

| Bundle File | File Size | Description |
|---|---:|---|
| `SURYOKTI_Full_Project.zip` | 89.79 MB | Complete, self-contained snapshot of the repository, excluding virtual environment (`.venv`) and temporary caches. Suitable for cloud upload, archiving, and remote sharing. |
| `SURYOKTI_All_Datasets.zip` | 31.97 MB | Consolidated archive of all 18 unique CSV datasets across Kaggle sources, NREL PVDAQ telemetry, and candidate aggregations. |

---

## 3. Duplicate Candidate Management

In accordance with strict safety invariants, files that share identical content or hashes across historical directories are preserved and documented in [`duplicate_candidates/DUPLICATE_FILES_INVENTORY.md`](duplicate_candidates/DUPLICATE_FILES_INVENTORY.md) to ensure existing workflows, legacy scripts, and external tools continue operating without disruption.

