# roi-generator

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
