# 🗺️ Project Roadmap

## Version Strategy
- **v0.x** → MVP & iterative improvements (current)
- **v1.x** → Major expansion: More Iranian car brands
- **v2.x** → Advanced analytics dashboard & market insights

##### 📍 Latest Version: **v0.1** – [Try the Live App](https://huggingface.co/spaces/kiarash2077/pride_car_price_estimator)

---

## v0.x Series - MVP & Polish

### v0.1 (Current MVP)
- [x] Complete ML pipeline for Pride cars
- [x] Live Gradio app on Hugging Face
- [x] Basic model comparison & selection

### v0.2 - UI & Optimization
- [ ] Enhanced web app interface (better Persian UX)
- [ ] Improved error handling and validation
- [ ] Performance optimizations for faster predictions

### v0.3 - Data Quality
- [ ] Multi-source data collection (additional websites)
- [ ] Automated data validation pipeline
- [ ] Enhanced outlier detection
- [ ] Historical data tracking

---

## v1.x Series - Brand Expansion 🚗

### Core Objective: Cover 70%+ of Iranian streets

### Phase 1: IKCO Family (Priority 1)
- [ ] **Samand** (all variants: LX, Soren, EL, etc.)
- [ ] **Peugeot Pars** & **Peugeot 405**
- [ ] **Dena/Dena+** & **Runna/Tara**
- [ ] **Rira** & other IKCO models

### Phase 2: Other Major Brands
- [ ] **Saipa Family**: Tiba, Saina, Quick
- [ ] **Renault**: Tondar90, Symbol, Megane
- [ ] **Hyundai/Kia**: Common models in Iranian market(Possible)
- [ ] **Toyota**: Popular models available in Iran(Possible)

### Technical Requirements:
- Multi-brand database architecture
- Unified feature encoding system
- Brand-specific model training or unified model approach
- Scalable scraping infrastructure

---

## v2.x Series - Market Intelligence Dashboard 📊

### Core Objective: Advanced analytics & insights platform

### Data Analysis Features:
- [ ] **Time-based Price Analysis**
  - Price trends by month/year
  - Brand comparison over time
  - Market movement forecasting

- [ ] **Comparative Analysis Dashboard**
  - Compare car brands by average price
  - Feature distribution across brands
  - Trim-level price comparisons
  - Fuel type & transmission analysis

- [ ] **Advanced Filter & Search**
  - Price range filtering with real-time results
  - Multi-criteria search (brand + year + mileage)
  - Saved searches and alerts
  - Export filtered results to CSV

- [ ] **Market Insights**
  - Most/least expensive brands
  - Price per age analysis
  - Regional price variations (if data available)
  - Popular configurations heatmap

---

### ♻️ Technical Implementation:
- **Backend**: FastAPI or Streamlit for dashboard
- **Frontend**: Interactive charts (Plotly, Altair)
- **Database**: Optimized for analytical queries
- **Export**: PDF/Excel report generation
---
**Note**: This roadmap is flexible and will adapt based on user feedback, market changes, and technical discoveries during implementation. Community suggestions are welcome via GitHub Issues.
