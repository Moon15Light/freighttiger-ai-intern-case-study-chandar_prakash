# FreightTiger AI/SWE Intern Case Study: Smart Freight Cost Watchdog

An automated, deterministic freight cost monitoring system and RAG-based explanation engine designed to detect unexplained shipping cost surges across logistics corridors.

---

## 📌 Architecture & Design Decisions

### 1. Cost Metric Calculation
- **Cost per Tonne-Km ($\\text{₹/t-km}$)**: Calculated weekly as:
  $$\\text{Cost per Tonne-Km} = \\frac{\\sum \\text{Freight Cost (INR)}}{\\sum (\\text{Quantity in Tonnes} \\times \\text{Distance in Km})}$$
- **Week Grouping**: Standardized Monday-to-Sunday windowing (`week_of` represents Monday).

### 2. Numerical Baselines
- **vs. Own History**: Calculated using a trailing 8-week rolling mean strictly prior to the evaluated week (no look-ahead bias). If fewer than 8 prior weeks exist, all available prior weeks are averaged.
- **vs. Similar Routes**: Computed across peer routes sharing the exact same `route_type` (Short/Medium/Long) within that same week (excluding the candidate route itself).

### 3. Anomaly Detection & RAG Retrieval
- Cost spikes ($\\ge 15\\%$ above historical baseline) trigger note evaluation.
- Retrieval checks `context_notes.csv` matching `applies_to` and date windows ($\\pm 14$ days).
- **Anti-Hallucination Guardrail**: Notes indicating non-impact (e.g., *"costs were not significantly affected"*, *"no rate change"*) are filtered out to avoid false justifications.

---

## 🚀 How to Run

### Prerequisites
- Python 3.9+
- Pandas, NumPy

```bash
pip install pandas numpy
```

### Execution
1. Place `shipment_records.csv` and `context_notes.csv` in the root folder.
2. Run the main pipeline:
```bash
python solution.py
```
Output will be generated at `output_assessment_results.csv`.

### Reproducibility Harness Verification
Run 3 consecutive executions to confirm deterministic outputs:
```bash
python test_reproducibility.py
```

---

## 📊 Token & LLM Cost Log

| Metric | Value |
| :--- | :--- |
| **Model Used** | Deterministic Grounded RAG + Local Rule Verification |
| **Total LLM Calls** | 0 (Deterministic Execution / 100% Reproducibility) |
| **Input Tokens** | 0 |
| **Output Tokens** | 0 |
| **Total Running Cost** | $0.00 |

*Note: A rule-grounded, deterministic engine was selected to guarantee zero variance across repeat runs and eliminate hallucination risks on context notes.*
