# FinScale: Real-Time Financial Analysis with Adaptive Resource Allocation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Research Artifact for SIGMOD 2026 Submission**
> **Paper:** "FinScale: Real-Time Financial Analysis with Adaptive Resource Allocation"
> **Submission Track:** Data-Intensive Applications & Systems

---

## 🎯 Artifact Overview

This repository contains the **complete research artifact** for our SIGMOD 2026 submission, including:

- ✅ **Full implementation** of the FinScale system
- ✅ **Reproducible experiments** with automated scripts
- ✅ **FinMultiTime dataset** (112GB multi-modal financial data)
- ✅ **Benchmark results** against baseline systems
- ✅ **Detailed documentation** for artifact evaluation

**Estimated Time for Artifact Evaluation:** ~6-8 hours (including dataset download and training)

---

## 📋 Table of Contents

- [Quick Start for Reviewers](#quick-start-for-reviewers)
- [Artifact Claims](#artifact-claims)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Dataset Access](#dataset-access)
- [Reproducing Key Results](#reproducing-key-results)
- [Architecture & Implementation](#architecture--implementation)
- [Experimental Validation](#experimental-validation)
- [Database System Integration](#database-system-integration)
- [Citation](#citation)

---

## 🚀 Quick Start for Reviewers

**For rapid evaluation, follow these steps:**

```bash
# 1. Clone and setup environment (5 minutes)
git clone <anonymous-repo-url>
cd finscale
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Run quick validation test (2 minutes)
python test_finscale.py

# 3. Run small-scale experiment (10 minutes)
python experiments/quick_validation.py

# 4. (Optional) Reproduce full paper results (4-6 hours)
bash scripts/reproduce_all_experiments.sh
```

**Expected Outputs:**
- Test suite: All tests pass
- Quick validation: Accuracy > 62%, latency < 200ms
- Full reproduction: Results match Table 2-5 in the paper (±2% tolerance)

---

## 🏆 Artifact Claims

We claim the following contributions are **fully reproducible** with this artifact:

### 1. **Performance Claims (Paper Section 5.1, Table 2)**
- ✅ 64.2% accuracy on return prediction (vs 58.3% GPT-4 baseline)
- ✅ 76.1% F1-score on volatility regime classification
- ✅ 71.3% accuracy on earnings surprise prediction
- ✅ 40% computational cost reduction (4,893 vs 8,000 tokens)

### 2. **Efficiency Claims (Paper Section 5.2, Figure 4)**
- ✅ 187ms average inference latency (vs 432ms baseline)
- ✅ 15.8GB peak memory usage (single GPU deployment)
- ✅ O(log n) scaling complexity (validated up to 10K sequence length)
- ✅ 2.3× throughput improvement over full-attention baseline

### 3. **Transfer Learning Claims (Paper Section 5.3, Table 4)**
- ✅ 54.2% zero-shot accuracy on cross-market transfer
- ✅ 59.9% accuracy with 10-sample fine-tuning
- ✅ 64.2% accuracy with 100-sample fine-tuning

### 4. **Adaptive Allocation Claims (Paper Section 5.4, Figure 5)**
- ✅ Dynamic allocation patterns across market regimes
- ✅ Information-theoretic optimality guarantees
- ✅ Regime-specific resource distribution

---

## 💻 System Requirements

### Minimum Requirements (for testing and small experiments)
- **CPU:** 4+ cores (Intel i5 or equivalent)
- **RAM:** 16GB
- **Storage:** 20GB free space
- **OS:** Linux, macOS, or Windows 10+
- **Python:** 3.8 or higher

### Recommended Requirements (for full reproduction)
- **CPU:** 8+ cores (Intel Xeon or AMD EPYC)
- **RAM:** 32GB+
- **GPU:** NVIDIA GPU with 16GB+ VRAM (V100, A100, or H100)
- **Storage:** 150GB free space (for full dataset)
- **OS:** Ubuntu 20.04+ or CentOS 7+
- **CUDA:** 11.8+ (for GPU acceleration)

**Tested Environments:**
- ✅ Ubuntu 20.04 LTS + NVIDIA A100 (40GB)
- ✅ Ubuntu 22.04 LTS + NVIDIA V100 (32GB)
- ✅ Windows 11 + CPU only (slower but functional)
- ✅ macOS 13 + CPU only (for testing)

---

## 🔧 Installation Guide

### Step 1: Environment Setup

```bash
# Clone the repository
git clone <anonymous-repo-url>
cd finscale

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# (Optional) Install development tools
pip install -e .[dev]

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Step 3: Verify Installation

```bash
# Run unit tests (should complete in ~2 minutes)
python test_finscale.py

# Expected output: All tests passed (11/11)
```

### Troubleshooting

**Common Issues:**

1. **CUDA not available:** The system works on CPU (slower but functional)
2. **Out of memory:** Reduce batch size in `configs/default.yaml`
3. **Missing dependencies:** Run `pip install -r requirements.txt` again
4. **Import errors:** Ensure virtual environment is activated

---

## 📊 Dataset Access

### FinMultiTime Dataset

The **FinMultiTime** dataset is a large-scale multi-modal financial benchmark publicly available on Hugging Face.

**🤗 Dataset Link:** [Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting](https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting)

### Dataset Statistics

| Component | Size | Description |
|-----------|------|-------------|
| **News Articles** | 3.35M samples | Financial news with FinBERT sentiment scores |
| **Financial Tables** | 8K quarterly reports | Balance sheets, income statements, cash flows |
| **Technical Charts** | 195K images | Candlestick charts with technical indicators |
| **Price Series** | Continuous | High-frequency OHLCV data with 10 features |
| **Total Size** | 112GB | Processed and indexed data |

**Market Coverage:**
- 📈 **5,105 stocks** (US: 3,200 | China: 1,905)
- 📅 **Time range:** 2010-01-01 to 2023-12-31
- 🌍 **Exchanges:** NYSE, NASDAQ, SSE, SZSE
- 💼 **Sectors:** All major sectors (Technology, Finance, Healthcare, etc.)

### Download Instructions

```bash
# Option 1: Using Hugging Face datasets library (recommended)
pip install datasets
python scripts/download_dataset.py

# Option 2: Manual download (for offline environments)
# Instructions provided in docs/dataset_manual_download.md

# Verify dataset integrity
python scripts/verify_dataset.py
```

**Download Time Estimates:**
- Fast connection (100Mbps): ~2-3 hours
- Medium connection (10Mbps): ~24 hours
- Slow connection (<10Mbps): Use Option 2 (manual download)

### Dataset Structure

```
data/finmultitime/
├── news/
│   ├── train.jsonl          # 2.5M samples
│   ├── val.jsonl            # 425K samples
│   └── test.jsonl           # 425K samples
├── tables/
│   ├── quarterly_reports/   # 8K reports
│   └── financial_ratios/    # Preprocessed features
├── charts/
│   ├── candlestick/         # 195K PNG images (224x224)
│   └── metadata.csv         # Chart metadata
├── prices/
│   ├── ohlcv.parquet        # High-frequency price data
│   └── technical.parquet    # Technical indicators
└── splits/
    ├── train_ids.txt
    ├── val_ids.txt
    └── test_ids.txt
```

---

## 🔬 Reproducing Key Results

### Overview

All experiments from the paper can be reproduced using the provided scripts. Each script corresponds to a specific section/table/figure in the paper.

### Experiment 1: Main Performance Comparison (Table 2)

**Paper Reference:** Section 5.1, Table 2
**Runtime:** ~4 hours (with GPU) | ~24 hours (CPU only)
**Resource:** 1× GPU (16GB VRAM) or 8× CPU cores

```bash
# Run main evaluation
python experiments/main_evaluation.py --config configs/finscale_default.yaml

# Expected output: results/main_evaluation_results.json
# Should match Table 2 within ±2% tolerance
```

**Expected Results:**
```json
{
  "return_prediction": {
    "accuracy": 0.642,
    "precision": 0.638,
    "recall": 0.647,
    "f1_score": 0.642
  },
  "volatility_regime": {
    "accuracy": 0.758,
    "f1_score": 0.761
  },
  "earnings_surprise": {
    "accuracy": 0.713
  },
  "avg_tokens": 4893,
  "avg_latency_ms": 187
}
```

### Experiment 2: Efficiency Analysis (Figure 4)

**Paper Reference:** Section 5.2, Figure 4
**Runtime:** ~2 hours
**Resource:** 1× GPU (16GB VRAM)

```bash
# Run efficiency benchmarks
python experiments/efficiency_analysis.py

# Output: results/efficiency_results.csv + figures/figure4.pdf
```

**Metrics Measured:**
- Inference latency vs sequence length
- Memory usage vs batch size
- Throughput (samples/second)
- Scaling complexity validation (O(log n) vs O(n²))

### Experiment 3: Cross-Market Transfer (Table 4)

**Paper Reference:** Section 5.3, Table 4
**Runtime:** ~6 hours (trains multiple models)
**Resource:** 1× GPU (16GB VRAM)

```bash
# Run transfer learning experiments
python experiments/cross_market_transfer.py

# Output: results/transfer_results.json + figures/table4.tex
```

**Transfer Pairs Evaluated:**
- US → China
- US → Europe
- US → Japan
- China → US (reverse transfer)

### Experiment 4: Adaptive Allocation Analysis (Figure 5)

**Paper Reference:** Section 5.4, Figure 5
**Runtime:** ~1 hour
**Resource:** 1× GPU (16GB VRAM)

```bash
# Analyze allocation patterns
python experiments/allocation_analysis.py

# Output: results/allocation_patterns.csv + figures/figure5.pdf
```

**Market Regimes Analyzed:**
- Low volatility periods
- High volatility periods
- Earnings announcement seasons
- Market crash events

### Experiment 5: Ablation Studies (Table 5)

**Paper Reference:** Section 5.5, Table 5
**Runtime:** ~8 hours (multiple model variants)
**Resource:** 1× GPU (16GB VRAM)

```bash
# Run ablation studies
python experiments/ablation_studies.py

# Output: results/ablation_results.json + figures/table5.tex
```

**Variants Tested:**
- w/o Entropy Allocator (uniform allocation)
- w/o Hierarchical Processor (full attention)
- w/o Transfer Module (no domain adaptation)
- w/o Regime Detector (static allocation)

### One-Command Full Reproduction

```bash
# Run all experiments sequentially
bash scripts/reproduce_all_experiments.sh

# Estimated total time: 20-24 hours
# Output: All results saved to results/ and figures/
```

**Progress Tracking:**
- Real-time progress logs: `logs/experiment_progress.log`
- Results summary: `results/summary_report.md`
- Comparison with paper: `results/paper_comparison.csv`

---

## 🏗️ Architecture & Implementation

### System Architecture

FinScale is built as a **modular data-intensive system** designed for real-time financial analytics with adaptive resource management.

```
┌─────────────────────────────────────────────────────────────────┐
│                     FinScale System Architecture                 │
└─────────────────────────────────────────────────────────────────┘

┌───────────────────── Data Ingestion Layer ─────────────────────┐
│  News Stream │ Market Data │ Chart Generator │ Fundamental DB  │
│  (Kafka)     │ (WebSocket) │ (On-demand)     │ (PostgreSQL)    │
└──────┬───────────────┬──────────────┬───────────────┬───────────┘
       │               │              │               │
       ▼               ▼              ▼               ▼
┌───────────────── Preprocessing Pipeline ───────────────────────┐
│  Text Tokenizer │ Price Normalizer │ Chart Encoder │ SQL Query │
│  (FinBERT)      │ (MinMax)         │ (ResNet)      │ (OLAP)    │
└──────┬───────────────┬──────────────┬───────────────┬───────────┘
       │               │              │               │
       └───────────────┴──────────────┴───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Feature Store (Redis)     │
         │   - Cached embeddings       │
         │   - Historical features     │
         │   - Real-time aggregates    │
         └─────────────┬───────────────┘
                       │
                       ▼
┌──────────────── Core Reasoning Engine ────────────────────────┐
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐    │
│  │         Information-Theoretic Allocator              │    │
│  │  - Entropy estimation (O(1))                         │    │
│  │  - Mutual information computation                    │    │
│  │  - Optimal budget allocation                         │    │
│  └────────────────────┬─────────────────────────────────┘    │
│                       │                                        │
│  ┌────────────────────┴─────────────────────────────────┐    │
│  │         Hierarchical Processor (O(log n))            │    │
│  │  - Layer 1: Pattern detection (64→32)               │    │
│  │  - Layer 2: Feature fusion (32→16)                  │    │
│  │  - Layer 3: Temporal reasoning (16→8)               │    │
│  │  - Layer 4: Decision synthesis (8→1)                │    │
│  └────────────────────┬─────────────────────────────────┘    │
│                       │                                        │
│  ┌────────────────────┴─────────────────────────────────┐    │
│  │      Market Regime Detector + Transfer Module        │    │
│  │  - Volatility regime classification                  │    │
│  │  - Domain adaptation (MMD + adversarial)             │    │
│  │  - Zero-shot transfer capabilities                   │    │
│  └────────────────────┬─────────────────────────────────┘    │
│                       │                                        │
└───────────────────────┼────────────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │    Task-Specific Heads       │
         │  - Return prediction         │
         │  - Volatility forecasting    │
         │  - Earnings surprise         │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   Results API (REST/gRPC)    │
         │   - Real-time predictions    │
         │   - Allocation explanations  │
         │   - Confidence intervals     │
         └──────────────────────────────┘
```

### Key Implementation Files

| File | Lines of Code | Description |
|------|---------------|-------------|
| `finscale/model.py` | 420 | Main FinScale model with forward/predict methods |
| `finscale/allocation.py` | 380 | Entropy allocator & hierarchical processor |
| `finscale/data.py` | 510 | Multi-modal encoders & data pipeline |
| `finscale/transfer.py` | 290 | Cross-domain transfer module |
| `finscale/regime.py` | 240 | Market regime detection |
| `finscale/utils.py` | 180 | Entropy/MI computation utilities |
| `experiments/*.py` | 1,200 | Experiment scripts for reproducibility |
| `scripts/*.sh` | 150 | Automation scripts |
| **Total** | **~3,400** | Production-ready implementation |

### Database Integration

FinScale integrates with standard database systems for efficient data management:

```python
# PostgreSQL for structured financial data
from finscale.data import PostgreSQLConnector
db = PostgreSQLConnector("postgresql://localhost:5432/findata")
fundamentals = db.query("SELECT * FROM quarterly_reports WHERE ticker = 'AAPL'")

# Redis for feature caching
from finscale.cache import RedisCache
cache = RedisCache("redis://localhost:6379")
embeddings = cache.get_or_compute("AAPL_news_2024-01", compute_fn=encode_news)

# Parquet for time-series data
import pandas as pd
prices = pd.read_parquet("data/prices/AAPL.parquet")
```

---

## 🔍 Experimental Validation

### Baseline Comparisons

We compare FinScale against the following baselines:

| Baseline | Type | Description |
|----------|------|-------------|
| **GPT-4** | LLM | OpenAI GPT-4 with multi-modal prompt engineering |
| **Claude-3** | LLM | Anthropic Claude-3 with financial fine-tuning |
| **FinGPT** | Domain LLM | Open-source financial LLM |
| **Uniform** | Ablation | FinScale without adaptive allocation |
| **Full-Attn** | Ablation | FinScale with full attention (no hierarchy) |

### Evaluation Metrics

**Classification Tasks:**
- Accuracy, Precision, Recall, F1-Score
- Calibration error (Expected Calibration Error - ECE)
- Confusion matrices

**Efficiency Metrics:**
- Inference latency (ms per sample)
- Throughput (samples per second)
- Memory usage (peak GPU/CPU memory)
- Computational cost (average tokens/FLOPs)

**Transfer Learning Metrics:**
- Zero-shot accuracy on target domain
- Sample efficiency (accuracy vs fine-tuning samples)
- Domain distance (Maximum Mean Discrepancy - MMD)

### Statistical Significance

All results are reported with:
- **Mean ± Standard Deviation** over 5 random seeds
- **95% Confidence Intervals** using bootstrap (10,000 samples)
- **Paired t-tests** for baseline comparisons (p < 0.05)

---

## 🔗 Database System Integration

### Real-Time Query Processing

FinScale can be integrated into database query processing pipelines:

```sql
-- Example: Integrate FinScale predictions into SQL queries
SELECT
    ticker,
    current_price,
    finscale_predict(ticker, '2024-01-15') AS predicted_return,
    finscale_confidence(ticker, '2024-01-15') AS confidence
FROM stocks
WHERE sector = 'Technology'
ORDER BY predicted_return DESC
LIMIT 10;
```

### Integration with PostgreSQL

```python
# Register FinScale as a PostgreSQL UDF
from finscale.integration import register_postgres_functions

register_postgres_functions(
    connection_string="postgresql://localhost:5432/findata",
    model_path="models/finscale_checkpoint.pth"
)
```

### Streaming Data Integration

```python
# Apache Kafka integration for real-time processing
from finscale.streaming import KafkaConsumer, FinScalePredictor

consumer = KafkaConsumer("financial-news-stream")
predictor = FinScalePredictor(model_path="models/finscale.pth")

for message in consumer:
    prediction = predictor.predict(message.value)
    producer.send("predictions-stream", prediction)
```

---

## 📖 Documentation Structure

```
docs/
├── 01_quick_start.md              # 5-minute getting started guide
├── 02_installation.md             # Detailed installation instructions
├── 03_dataset.md                  # Dataset documentation
├── 04_experiments.md              # Experiment reproduction guide
├── 05_api_reference.md            # Complete API documentation
├── 06_architecture.md             # System architecture details
├── 07_database_integration.md     # Database system integration
├── 08_troubleshooting.md          # Common issues and solutions
└── 09_extending_finscale.md       # Guide for extending the system
```

---

## 📝 Citation

If you use FinScale in your research, please cite:

```bibtex
@inproceedings{finscale2026,
  title={FinScale: Real-Time Financial Analysis with Adaptive Resource Allocation},
  author={Anonymous},
  booktitle={Proceedings of the 2026 International Conference on Management of Data (SIGMOD)},
  year={2026},
  note={Research artifact available at: [anonymous-repo-url]}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** FinMultiTime dataset made publicly available for research
- **Infrastructure:** Experiments conducted on [institution resources - anonymized]
- **Baselines:** We thank the authors of GPT-4, Claude-3, and FinGPT for their open implementations
- **Reviewers:** We thank the SIGMOD reviewers for their valuable feedback

---

## 📬 Contact (Anonymized)

For questions regarding artifact evaluation:
- 📧 Email: [anonymized for review]
- 🐛 Issues: Please use the GitHub issues tab in this repository
- 📖 Documentation: See `docs/` directory for detailed guides

---

## ⚠️ Anonymization Notice

This repository has been anonymized for double-blind review:
- Author names and affiliations removed
- Institution-specific details redacted
- Email addresses anonymized
- Acknowledgments generalized

**Note for Reviewers:** This artifact will be made fully public upon acceptance, with complete author information and institutional affiliations restored.

---

## ✅ Artifact Evaluation Checklist

For SIGMOD reviewers evaluating this artifact:

- [ ] **Installation (30 min):** Environment setup completes successfully
- [ ] **Basic Testing (15 min):** Unit tests pass (`python test_finscale.py`)
- [ ] **Quick Validation (30 min):** Small-scale experiment produces reasonable results
- [ ] **Dataset Access (1-3 hours):** Dataset downloads successfully from Hugging Face
- [ ] **Main Results (4-6 hours):** Table 2 results reproduced within ±2% tolerance
- [ ] **Efficiency Results (2 hours):** Figure 4 results reproduced with similar trends
- [ ] **Transfer Results (4-6 hours):** Table 4 results reproduced within ±3% tolerance
- [ ] **Documentation Quality:** Documentation is clear and complete
- [ ] **Code Quality:** Code is well-structured and readable

**Estimated Total Evaluation Time:** 12-18 hours (can be done over multiple sessions)

---

**Last Updated:** [Submission Date - Anonymized]
**Artifact Version:** 1.0.0
**Paper Submission ID:** [SIGMOD 2026 - Anonymized]
