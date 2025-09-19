import subprocess
import sys
from pathlib import Path

def main():
    cmd = [
        "deepspeed", "--num_gpus=1",
        "train.py",
        "--deepspeed_config=ds_config.json"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()