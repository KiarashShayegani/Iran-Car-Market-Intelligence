# Changelog

All notable changes to the Iran Car Market Intelligence project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

> ### [v0.2] - 2026-12-2
> #### Minimal repo commits
> **Added:**
>  - System architecture diagram
>  - GUI screenshots
>  - Additional README.md for folders
>  - URL and tiny text fixes

---

## [v0.1 - MVP] - 2026-01
### Initial Minimum Viable Product Release

### Added
- **Complete ML Pipeline**: Web scraping → data cleaning → model training → deployment
- **Four ML Models**: Random Forest, XGBoost, Neural Network, Linear Regression comparison
- **Live Web App**: Gradio interface deployed on Hugging Face Spaces
- **MySQL Database**: Structured storage with Persian language support

### Features
- Price estimation for Pride cars (Saipa)
- Iranian-specific feature processing (کارکرده mileage, Persian trim names)
- Real-time predictions with seven input features
- Model evaluation and selection framework

### Technical Highlights
- **Best Model**: Random Forest (R²: 0.872)
- **Dataset**: 580+ listings from bama.ir
- **Deployment**: Hugging Face Spaces
- **Architecture**: Modular Python codebase

### Current Scope (MVP)
- Single car brand (Pride only)
- Single data source
- Basic web interface


---

## [v1.x - Planned] - Q1 2026
### Multi-Source Data Integration

### Planned Additions
- **New Brand Support**
  - Iran Khodro (IKCO) car models: Samand, Peugeot Pars, Dena, etc.
  - Expanded database schema for multi-brand support
  - Brand-specific feature engineering
  - Comparative analysis across brands

---

## Versioning Note
- **MVP**: Initial proof-of-concept with Pride cars only
- **v0.x**: Development versions with incremental features
- **v1.x**: First stable release with multiple Iranian car brands
- **v2.x**: Major feature releases and market analysis dashboard

---

> *For detailed future plans, see [ROADMAP.md](ROADMAP.md)*
> 
> *For project background and vision, see [PROJECT_VISION.md](PROJECT_VISION.md)*
