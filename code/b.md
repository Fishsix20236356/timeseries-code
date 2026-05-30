
**Context & File Overview**
Detailed design workflow for a MOMENT + Chronos-2 Joint Training Framework, integrating MOMENT as a robust universal feature extractor and Chronos-2 as a decoder/predictor capable of probabilistic forecasting and in-context learning.

1. **Context & File Structure** The current working directory contains:

* **`data.md`**: Documentation for the data preprocessing workflow.
* **`project.md`**: Detailed instructions for the training process.
* **`ref.txt`**: Reference code and guidelines.
* **`chronos_finetune.py`**: Code for independent LoRA fine-tuning of the **Chronos2** model.
* **`Representation.py`**: Code for generating embeddings using the **Moment** model.
* **`./json/`**: Directory containing time series data files named {code}.json.
* **`info2.xlsx`**: Metadata mapping. Columns: code (matches JSON filename) and chnName (Chinese feature name).




2. **Data Specifications**
Format: JSON files structured as [{"time": "2023-11-17 22:00:00", "value": 5.64187}, ...]
Variable Mapping: Use info2.xlsx to map variable codes to names.
Variable Types:
Target Variables: ["铁水温度", "铁水Si", "炉渣二元碱度"] (Forecast these)
Covariates: ["喷煤量", "冷水流量", "氧气流量", "风压力"] (Use as conditional context)

Use HuggingFace AutonLab/MOMENT-1-large model


**Objective**
You are required to implement the complete code for both data preprocessing (guide in data.md)and model training of **MOMENT + Chronos-2 联合训练框架**(project.md).

**Requirements**
Before writing the code, please provide a comprehensive **Execution Plan**.

