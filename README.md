## Project layout

```text
SwinScore/
├── radiology/                                      # CT imaging data paarent folder
│   ├── data_table.csv                              # clinical data
│   ├── data_table_test.csv                         # clinical data + computed tumor bounding boxes coordinates (obtained from Step 1 below)
│   ├── 11409285/
│   │   ├── image.nii.gz                            # CT scan
│   │   ├── label.nii.gz                            # binary tumor mask
│   │   └── node.nii.gz                             # binary node mask
│   ├── 18781944/
│   │   ├── image.nii.gz
│   │   ├── label.nii.gz
│   │   └── node.nii.gz
│   └── ...                                         # one folder per patient; folder name should appear in the "radiology_folder_name" column in "data_table_test.csv"
│
├── checkpoints/
│   └── raw_images/
│       └── fused_attention_classification_50_0.001/
│           └── fused_attention_classification_50_0.001_best_val_cindex.pt
│                                                   # model checkpoint
│
└── swin_radiomics/                                 # root folder for code (run all commands from here)
    ├── main.py                                     # main scripts
    ├── datasets.py                                 # PyTorch Dataset
    ├── models.py                                   # swinT extractor + MLP head
    ├── losses.py                                   # weighted BCE loss
    ├── utils.py                                    # metrics, cropping, windowing, optimizer
    ├── parameters.py                               # argument parser and defaults
    ├── preprocessing.py                            # computes tumor, node bounding boxes coordinates from masks
    ├── swintransformer.py                          # Swin Transformer building blocks
    └── requirements.txt                            # Swin Transformer building blocks (unused in current pipeline)
```

---

## Running the pipeline

> Run all commands from inside `swin_radiomics/`.

### Step 0: Prepare the clinical sheet

Make sure you have a file called `data_table.csv` at:

```text
.\SwinScore\radiology\
```

This file is the clinical sheet on the test cohort. It should contain **one row per patient** with the following required columns:

- `radiology_folder_name`: folder name for each patient within the radiology folder
- `DFS_3years`: binary values of `0` (censored or had event after 36 months) or `1` (had event within 36 months)
- `DFS`: continuous values representing survival (i.e. OS, DFS, LRF, etc.) in months
- `DFS_censor`: `0` (censored) or `1` (event)

---

### Step 1: Compute bounding boxes (one-time setup)

> **Important:** You must have `data_table.csv` (clinical sheet from the previous step) inside:
>
> ```text
> .\SwinScore\radiology\
> ```

This step generates `data_table_test.csv` with the following new columns:

- `X_min_tumor`, `Y_min_tumor`, `X_max_tumor`, `Y_max_tumor`, `Z_min_tumor`, `Z_max_tumor`
- `X_min_lymph`, `Y_min_lymph`, `X_max_lymph`, `Y_max_lymph`, `Z_min_lymph`, `Z_max_lymph`

**Command**

```bash
python preprocessing.py
```

---

### Step 2: Inference

This step reads:

- `data_table_test.csv`
- the CT images within each patient subfolder
- the checkpoint file

and performs inference.

**Command**

```bash
python main.py
```
