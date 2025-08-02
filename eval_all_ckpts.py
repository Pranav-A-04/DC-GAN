import os
import subprocess

ckpt_dir = "checkpoints"
eval_script = "eval.py"
csv_path = "fid_scores.csv"

# Optional: Clear the old CSV file first (uncomment if needed)
# if os.path.exists(csv_path):
#     os.remove(csv_path)

ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pth")]
ckpts.sort()

for ckpt in ckpts:
    ckpt_path = os.path.join(ckpt_dir, ckpt)
    print(f"Running FID eval for: {ckpt_path}")
    subprocess.run(["python", eval_script, "--ckpt", ckpt_path, "--csv", csv_path])