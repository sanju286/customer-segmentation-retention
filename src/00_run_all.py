"""Run the full pipeline: python src/00_run_all.py"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["01_data_acquisition.py", "02_rfm_segmentation.py",
         "03_churn_modeling.py", "04_clv_estimation.py", "05_business_insights.py"]

for step in STEPS:
    print(f"\n{'=' * 60}\n>>> {step}\n{'=' * 60}")
    r = subprocess.run([sys.executable, os.path.join(HERE, step)])
    if r.returncode != 0:
        print(f"Pipeline stopped at {step} (exit {r.returncode})")
        sys.exit(r.returncode)
print("\nPipeline complete. See outputs/reports/BUSINESS_REPORT.md")
