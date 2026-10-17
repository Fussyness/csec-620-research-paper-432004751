# Data

**Do not commit datasets to Git.** They are large, may be license-restricted, and bloat history.

Instead:

1. Document exactly where the data comes from (name, version, URL, date accessed).
2. Provide a script or one-liner to download it into this folder.
3. Reference the file(s) via `config.yaml` (`data.source: csv`, `data.csv_path: data/your_file.csv`).

## This project's dataset

- **Name / source:** _e.g., CIC-IoT2023 — https://www.unb.ca/cic/datasets/iotdataset-2023.html_
- **Download:** _command or steps_
- **Target column:** _e.g., `label`_
- **Notes:** _class balance, preprocessing, known label issues_

The template ships with a **synthetic** dataset (`data.source: synthetic`) so `make reproduce`
runs before you wire up real data.
