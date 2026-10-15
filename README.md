# roi-generator

> [!IMPORTANT]
> **Archived (2026-10-15).** This code moved to [ai-gen-image-stats](https://github.com/kktsuji/ai-gen-image-stats) as the `roi_extraction` experiment ([#106](https://github.com/kktsuji/ai-gen-image-stats/pull/106)): `--mode catalog` and `--mode extract` replace `generate-roi.py`, and `--mode render` (`rendering.method: pseudo_rgb8_vstretch`) builds the pseudo-RGB images. The new code selects channels by file-name suffix instead of `glob` order and reproduces this repository's output pixel for pixel.

Prepare a python environment:

```bash
python --version
# 3.13.2

python -m venv vnevn
source vnevn/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Create `.env` file with the following content:

```
ROOT_DIR_PATH="your/path/to/root/dir"
OUT_DIR_PATH="your/path/to/output/dir"
```

Run the script:

```bash
python generate-roi.py
```
