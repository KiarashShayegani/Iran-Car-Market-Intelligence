# 🚗 Iran Car Market Intelligence

**An ML-powered car price estimation system for Iran's automotive market, featuring data scraping, model training, and deployment to provide intelligent price predictions beyond basic database queries.**

[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/kiarash2077/pride_car_price_estimator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-brightgreen)](https://www.python.org/downloads/)
[![Project Status](https://img.shields.io/badge/Status-MVP%20Complete-success)](https://github.com/KiarashShayegani/iranian-car-market-intelligence)

## 📖 Project Origin

> *"When navigating Iran's volatile car market, both buyers and sellers face a common challenge: determining a fair price for a vehicle. Existing platforms often rely on simple database queries that return minimum, maximum, and average prices—useful, but potentially missing complex patterns and vulnerable to outliers.*
>
>*This project began with a question: Could machine learning provide more nuanced price estimations by learning from actual market data? Starting with Pride cars (Iran's most common vehicle), I built a complete pipeline—from data collection to deployment—to explore whether ML models could offer a complementary approach to traditional price estimation methods."* //more explanations to be added * * * 

## 🔍 What This Project Is

- A **complete ML pipeline** for car price estimation, tailored for Iran's market
- A **data-driven approach** that learns from real listings on bama.ir
- A **modular system** with scraping, cleaning, training, and deployment components
- An **exploratory project** comparing multiple ML algorithms on automotive data
- A **practical application** deployed as an interactive web interface

## 🏗️ Architecture Overview

The system follows a structured pipeline:

```
📊 Data Collection → 🧹 Data Processing → 🧠 Model Training → 🚀 Deployment
      ↓                     ↓                     ↓                ↓
  Web Scraping        Cleaning &         4 ML Algorithms    Gradio Web App
  (bama.ir API)       Encoding           with Comparison    on Hugging Face
```

## 📈 Technical Implementation

### Data Pipeline
- **Scraping**: Extracts 580+ Pride car listings from bama.ir with Persian text handling
- **Cleaning**: Processes unique Iranian market features (کارکرده mileage, Persian trim names)
- **Database**: Stores structured data in MySQL with UTF-8 support for Persian characters
- **Encoding**: Custom ordinal encoding for Iranian-specific features (body_status hierarchy)

### Machine Learning Models
Four algorithms were trained and compared:

| Model | R² Score | Key Characteristics |
|-------|----------|---------------------|
| **Random Forest** | 0.872 | Best overall performance, handles non-linear relationships |
| **XGBoost** | 0.865 | Fast training, good with tabular data |
| **Neural Network** | 0.843 | Deep learning approach, more complex architecture |
| **Linear Regression** | 0.812 | Simple baseline model |

### Features Used
```
'name' → Car model name (encoded)
'trim' → Trim level (ordinal encoded)
'mileage' → Kilometers driven (imputed for 'کارکرده')
'fuel' → Fuel type (بنزینی/دوگانه سوز)
'transmission' → Transmission type (دنده ای/اتوماتیک)
'body_status' → Body condition (15-level hierarchy)
'age' → Car age (calculated from year)
```

## 🚀 Live Demo

Try the interactive price estimator:
- **Live App**: [Pride Car Price Estimator on Hugging Face](https://huggingface.co/spaces/kiarash2077/pride_car_price_estimator)
- **Current Scope**: Pride cars (Saipa) only
- **Features**: All input parameters with Persian interface support

## 📂 Repository Structure

```
iranian-car-market-intelligence/
│
├── src/                          # Source code
│   ├── data_pipeline/            # Scraping, cleaning, database operations
│   ├── ml/                       # Model training and evaluation
│   └── app/                      # Web application (Gradio)
│
├── notebooks/                    # Jupyter notebooks for experimentation
│   ├── exploration.ipynb         # Initial EDA and data analysis
│   └── modeling_experiments.ipynb # Model development and comparison
│
├── models/                       # Trained models and artifacts
│   └── v1/                       # Version 1 models
│       ├── random_forest_model.pkl
│       ├── input_scaler.pkl
│       ├── model_metadata.json
│       └── feature_importance.csv
│
├── docs/                         # Documentation
│   ├── methodology.md            # Technical methodology
│   ├── api.md                    # API documentation (future)
│   └── deployment.md             # Deployment guides
│
├── PROJECT_VISION.md             # Detailed project background and vision
├── ROADMAP.md                    # Future development plans
├── CHANGELOG.md                  # Version history
└── requirements.txt              # Python dependencies
```

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- MySQL (for full pipeline)
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/KiarashShayegani/iranian-car-market-intelligence.git
cd iranian-car-market-intelligence

# Install dependencies
pip install -r requirements.txt

# Run the web app locally
python src/app/gradio_app.py
```

### Run the Complete Pipeline
1. **Data Collection**: `python src/data_pipeline/scraper.py`
2. **Data Cleaning**: `python src/data_pipeline/cleaner.py`
3. **Database Setup**: `python src/data_pipeline/database.py`
4. **Model Training**: `python src/ml/trainer.py`
5. **Launch App**: `python src/app/gradio_app.py`

## 🎯 Current Status & Roadmap

### ✅ MVP Complete
- [x] Pride car data pipeline (scraping → cleaning → database)
- [x] Four ML models trained and evaluated
- [x] Gradio web interface deployed
- [x] Complete documentation

### 🔄 In Development
- [ ] Expand to additional Iranian car brands (Peugeot, Samand, etc.)
- [ ] Implement real-time data updates
- [ ] Add FastAPI backend for production use
- [ ] Create mobile-friendly interface

### 📋 Planned Features
- [ ] Multi-brand price estimation
- [ ] Historical price trends visualization
- [ ] Regional price variations (by Iranian city)
- [ ] API for third-party integration
- [ ] Docker containerization

## 🧪 Methodology Details

### Data Processing Highlights
- **Persian Text Handling**: Full support for UTF-8 and Persian character processing
- **Mileage Imputation**: Intelligent handling of "کارکرده" (used) vehicles using year-based grouping
- **Custom Encoding**: Domain-specific ordinal encoding for Iranian car features
- **Outlier Management**: IQR-based outlier detection and handling

### Model Selection Rationale
The Random Forest model was selected for deployment due to:
1. **Best performance** on test data (R² = 0.872)
2. **Feature importance** insights for interpretability
3. **Robustness** to outliers and noise in real-world data
4. **Reasonable training time** for potential periodic retraining

## 🤝 Contributing

This project welcomes contributions! Whether you're interested in:
- Adding support for more car brands
- Improving the ML models
- Enhancing the web interface
- Optimizing the data pipeline

Please check the [Contributing Guidelines](CONTRIBUTING.md) and feel free to open issues or submit pull requests.

## ⚠️ Important Notes

- **Data Source**: This project uses publicly available data from bama.ir for educational purposes
- **Scope Limitation**: Currently only supports Pride cars as an MVP
- **Market Specificity**: Designed specifically for Iran's unique automotive market conditions
- **Educational Purpose**: Primarily a demonstration of ML pipeline building for real-world problems

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sourced from [bama.ir](https://bama.ir)
- Built as a practical ML engineering project
- Special thanks to the open-source community for the tools and libraries used

---

**Disclaimer**: This is an educational project demonstrating ML pipeline development. Price estimations should be verified with multiple sources for real purchase/sale decisions.
