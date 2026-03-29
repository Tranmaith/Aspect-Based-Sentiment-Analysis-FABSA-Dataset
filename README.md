# Unified Aspect-Based Sentiment Analysis (ABSA) on the FABSA Corpus

*A Comparative Study of Sparse Linear Baselines, Recurrent Neural Architectures, and Transformer-based Transfer Learning.*

## 1. Project Abstract
This repository presents a rigorous, three-stage implementation of Aspect-Based Sentiment Analysis (ABSA) on the advanced Feedback ABSA (FABSA) dataset. We resolve the joint task of Aspect Category Detection (ACD) and Aspect Category Sentiment Classification (ACSC) by formulating it as a 36-way Multi-Label Classification problem.

**Research Significance:** Unlike standard sentiment analysis, this pipeline untangles "Semantic Packing" where a single review evaluates multiple aspects (e.g., Price vs Service) with potentially conflicting polarities. The report address the 52.80% multi-label density using a phased approach:

- Stage 1: Sparse Linear Baselines (SVM) with Information-Theoretic difficulty stratification.
- Stage 2: Recurrent Sequence Modelling (BiLSTM) with Custom Word Embeddings.
- Stage 3: Transformer Transfer Learning (DistilRoBERTa-Base). Optimised for high-throughput inference in CPU-constrained environments while retaining 95% of the performance of the full RoBERTa architecture.

## 2. Sequential Execution Workflow & State Persistence
The research pipeline is divided into three notebooks to manage computational overhead and ensure consistent evaluation across different model architectures. Follow the mandatory execution order below:

### 2.1 Phase 1: Notebook 1 (Baseline & SVM)
Action: Execute the complete cleaning pipeline, EDA, and TF-IDF vectorisation.

Critical Step: Run the final Artefact Export cell to "lock" the MultiLabelBinarizer (MLB) state and processed datasets.

Result: Generates baseline_results.joblib and processed_data.joblib in the ./data/processed/ directory.

### 2.2 Phase 2: Notebook 2 (BiLSTM Modelling)
Action: Initialise the BiLSTM architecture using the pre-processed tensors exported from Phase 1.

Critical Step: Execute the final evaluation audit to persist the bilstm_performance.csv and model weights.

Dependency: Uses the Phase 1 static data split to ensure fair "Cross-Model Comparison."

### 2.3 Phase 3: Notebook 3 (Transformer Fine-Tuning)
Action: Perform DistilRoBERTa-Base fine-tuning for joint ACD-ACSC.

Critical Step: Import the binarised labels directly from the Phase 1 exports to maintain label indexing integrity.

Validation: Run the post-hoc Threshold Calibration to resolve the Neutral class bottleneck.

Technical Rationale for Modular Execution
Memory Efficiency: Prevents RAM overflow in CPU-bound environments by separating TF-IDF and Transformer tensor operations.

State Consistency: Ensures the Joint Penalty Gap is measured against an immutable label distribution across all three paradigms.

Checkpointing: Validates that training failures in Phase 3 do not require a full re-run of Phase 1 EDA or Phase 2 sequential training.

## 3. Dataset & Information-Theoretic Profiling:
The FABSA dataset comprises 7,930 instances across 12 Aspect Categories and 3 Polarity states.

The "Difficulty Hierarchy" (KL Divergence)
The study quantifies "Shortcut Learning" risk by measuring the Kullback–Leibler (KL) Divergence of each aspect's sentiment distribution against a uniform prior:

Aspect Category                   DKL​            Classification                    Evaluation Priority
Staff support: Email             0.573           High-Bias (Easy)                  Recall (Detect rare signal)
Online experience: App           0.216           Low-Bias (Hard)                   Precision (Semantic resolution)
Overall Dataset                   -              Multi-Label                       Jaccard / Hamming Loss

## 4. Repository Architecture
The pipeline is designed for Linear Persistence: results from Notebook 1 are cached and utilised to "harden" subsequent deep learning stages.

## Project Structure

```
├── data/
│   ├── fabsa_*.csv            # Raw Financial Aspect-Based Sentiment Dataset
│   └── processed_results/     # "Locked" MLB state, Cleaned Text, and Tensors
├── results/
│   ├── models/                # .joblib (SVM) and .pt (BiLSTM/Transformer)
│   ├── reports/               # Master_performance_comparison.csv
│   └── plots/                 # Learning Dynamics & Task Decompositions
├── Notebook_1_SVM.ipynb       # Phase 1: EDA, Feature Engineering & Baseline Lock
├── Notebook_2_BiLSTM.ipynb    # Phase 2: Sequential Neural Modelling
└── Notebook_3_DeBERTa.ipynb   # Phase 3: Transformer Fine-tuning & Optimisation
```


## 5. Experimental Phases

- Phase 1: SVM Baseline (The Performance Floor)
    . Vectorisation: TF-IDF (1,2-grams) | max_features=5000 | min_df=2.
    . Architecture: One-vs-Rest LinearSVC with squared_hinge loss.
    . Optimisation: $C=0.1$ (Regularised to prevent n-gram memorisation).
    . Calibration: Class-specific decision thresholds derived from Val-set PR-curves.
    . Outcome: Established a Micro F1 of 0.478 and identified a 13.6% Coupling Penalty for the joint task.

- Phase 2: BiLSTM (Sequential Context)
    . Embedding: 100-dim learned vectors | 3,000-word core lexicon.
    . Architecture: Bi-directional LSTM (256 units) + Sigmoid Multi-head Output.
    . Regularisation: Post-embedding (0.3) and Hidden-state (0.4) Dropout.
    . Objective: Capture long-distance dependencies and negation (e.g., "not as good as expected").
    
- Phase 3: DistilRoBERTa (Disentangled Attention)
    . Model: microsoft/DistilRoBERTa-base.
    . Innovation: Uses a Relative Position Disentangled Attention mechanism.
    . Strategy: Differential fine-tuning (Freezing early layers) + AdamW with weight decay.
    . Coverage: 100% data retention (all sequences < 512 tokens).
    
## 6. Metric Priority Framework
To ensure research integrity, we reject "Accuracy" in favour of a stratified metric suite:
- Micro F1: Global efficacy.
- Macro F1: Robustness against the 3.64% Neutral class scarcity.
- Jaccard Score: Set-overlap accuracy for multi-label instances.
- Hamming Loss: Measures individual bit-error rate across the 36-label vector.

## 7. Installation & Setup
Environment Requirements
Python 3.11+CUDA 12.1 (Recommended for Notebooks 2 & 3)

Bash
# Clone and enter the directory
git clone https://github.com/Tranmaith/Aspect-Based-Sentiment-Analysis-FABSA-Dataset.git
cd Aspect-Based-Sentiment-Analysis-FABSA-Dataset

# Install dependencies
pip install -r requirements.txt

Dataset Loading

The notebooks implement an Automated Fallback Loader:
- Attempts to fetch jordiclive/FABSA via Hugging Face API.
- If API fails, automatically switches to local CSVs in ./data.7. 

Performance Benchmark (Final Results)

## 📊 Model Performance Comparison

| Model         | Micro F1 | Macro F1 | Jaccard |
| ------------- | :------: | :------: | :-----: |
| SVM Baseline  |  0.4780  |  0.4070  |  0.3460 |
| BiLSTM        |  0.5418  |  0.3892  |  0.4415 |
| DistilRoBERTa |  0.6234  |  0.4521  |  0.5098 |

### 7.1 Hardware-Agnostic Adaptation (CPU Optimisation)
A core technical challenge of this study was the execution of Transformer fine-tuning within an i386 emulated CPU environment (Apple Silicon/Virtualised Institutional Hardware). To resolve "Illegal Instruction" and "DummyObject" errors common in cross-architecture deep learning, the following optimisations were implemented:

Model Distillation: Pivoted from DeBERTa-v3 to DistilRoBERTa-Base. This reduced the parameter count from 184M to 82M, decreasing the memory footprint by ~55% without significant loss in Micro-F1.

Gradient Accumulation: To simulate a robust batch size of 16 on limited RAM, we implemented gradient_accumulation_steps=4 with a per_device_train_batch_size=4.

Vectorised Float32 Casting: A custom MultiLabelTrainer was engineered to force float32 label casting at the tensor level, bypassing type-mismatch overheads during the Binary Cross-Entropy loss calculation on CPU backends.

## 8. Implementation & Reproducibility Guide
 Model Acquisition (Transformers)
The pipeline utilizes the microsoft/deberta-v3-base model via the Hugging Face Hub.

Automatic Download: Upon the first execution of Notebook 3, the transformers library will automatically fetch the ~370MB model weights and SentencePiece tokenizer.

Cache Management: If the environment encounters a JSONDecodeError or OSError during the initial handshake, it is recommended to clear the local transformer cache to ensure integrity:

Bash
**Manual cache purge for institutional environments**
rm -rf ~/.cache/huggingface/hub/models--microsoft--deberta-v3-base

## 9. Dependency Synchronisation & Portability
### 9.1 Version Alignment (CPU-Stable Baseline)
To ensure the pipeline remains executable on standard institutional hardware without requiring GPU acceleration, the stack is locked to the following stable baseline:

PyTorch: 2.2.1 (CPU-Optimized Build)

Transformers: 4.38.2

Datasets: 2.18.0

### 9.2 Implementation Rationale
While newer versions of Transformers (5.x+) exist, they require PyTorch 2.4+ features (such as enhanced C++ hooks for Scaled Dot Product Attention). By synchronising the stack to the 4.37.x baseline, we maintain the integrity of the Disentangled Attention mechanism while ensuring the notebook remains fully executable on standard institutional hardware without requiring complex system-level OS updates.

### 9.3 Environment Sanitisation
To maintain the professional quality of the research output, all absolute local paths and redundant library warnings (e.g., Newly Initialised Weights) are suppressed in the final notebook execution. This ensures the Triple Model Performance Audit remains the focal point of the report.

## 10. Author & Citation
Author: Tran Mai Thuong Git @Tranmaith
Dataset Source: Hugging Face: jordiclive/FABSA
Task: Aspect-Based Sentiment Analysis (ABSA) for Business Intelligence.
