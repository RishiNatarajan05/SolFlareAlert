# FlareAlert - Solar Flare Prediction System

A comprehensive real-time solar flare prediction and monitoring system with machine learning capabilities, modern web interface, and automated data ingestion.

## 🌟 Features

### 🔮 **Advanced ML Predictions**
- **Hazard Ensemble Model** - XGBoost + Logistic Regression ensemble
- **Real-time Predictions** - 6-hour solar flare probability forecasts
- **51 Feature Engineering** - Comprehensive solar activity features
- **Model Performance** - AUC: 0.772, Precision: 0.502, Recall: 0.176

### 📊 **Real-time Monitoring**
- **Live Dashboard** - Animated risk gauges and real-time statistics
- **WebSocket Integration** - Instant updates and alerts
- **Solar Activity Tracking** - Flares, CMEs, and geomagnetic storms
- **Historical Analysis** - 7-day data visualization

### 🚀 **Modern Web Interface**
- **React Dashboard** - Beautiful, responsive UI with dark theme
- **Interactive Charts** - Real-time data visualization
- **Mobile Responsive** - Works on all devices
- **Smooth Animations** - Framer Motion powered transitions

### 🔔 **Smart Alert System**
- **Risk-based Alerts** - Configurable thresholds (MINIMAL, LOW, MEDIUM, HIGH)
- **Real-time Notifications** - WebSocket-powered instant alerts
- **Email & Webhook Support** - Multiple notification channels
- **Alert History** - Track and manage past alerts

<<<<<<< Updated upstream
- **Precision**: 50.2%
- **False Positive Rate**: 4.8%
- **Ensemble AUC**: 0.781
=======
### 🔄 **Automated Operations**
- **Data Ingestion** - NASA DONKI API integration
- **Model Retraining** - Automated model updates
- **Background Monitoring** - Continuous solar activity tracking
- **Health Checks** - System status monitoring
>>>>>>> Stashed changes

## 🏗️ Architecture

```
FlareAlert/
├── backend/                 # FastAPI Backend
│   ├── app/
│   │   ├── main.py         # FastAPI application with WebSocket
│   │   └── prediction_service.py  # ML prediction service
│   ├── database/
│   │   └── models.py       # SQLAlchemy models
│   ├── data/
│   │   ├── ingestion.py    # NASA DONKI data ingestion
│   │   └── features.py     # Feature engineering
│   └── models/             # Trained ML models
├── frontend/               # React Frontend
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/         # Page components
│   │   ├── services/      # API & WebSocket services
│   │   └── styles/        # Tailwind CSS styles
│   └── public/            # Static assets
└── scripts/               # ML training scripts
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **SQLAlchemy** - Database ORM
- **SQLite** - Lightweight database
- **WebSocket** - Real-time communication
- **XGBoost** - Machine learning
- **Pandas/NumPy** - Data processing

### Frontend
- **React 18** - Modern React with hooks
- **React Router** - Client-side routing
- **React Query** - Server state management
- **Tailwind CSS** - Utility-first styling
- **Framer Motion** - Animations
- **Lucide React** - Icons
- **Axios** - HTTP client

### ML/AI
- **Hazard Ensemble Model** - Custom ensemble approach
- **Feature Engineering** - 51 comprehensive features
- **Model Persistence** - Pickle-based model storage
- **Real-time Inference** - Fast prediction serving

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Git

### 1. Clone Repository
```bash
git clone <repository-url>
cd FlareAlert
```

### 2. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

### 4. Start Services

**Backend (Terminal 1):**
```bash
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend (Terminal 2):**
```bash
cd frontend
npm start
```

### 5. Access Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📡 API Endpoints

### Core Endpoints
- `GET /` - API information
- `GET /health` - Health check
- `POST /predict` - Solar flare prediction
- `GET /model-info` - Model information
- `GET /solar-activity` - Current solar activity
- `GET /historical-data` - Historical data (7 days)

### Management Endpoints
- `POST /ingest-data` - Trigger data ingestion
- `POST /retrain-model` - Retrain ML model
- `POST /alerts` - Configure alert settings

### Real-time
- `WS /ws` - WebSocket for real-time updates

## 🎯 Key Features in Detail

### Risk Assessment
- **MINIMAL** (0-20%): Safe conditions
- **LOW** (20-40%): Low risk, monitor
- **MEDIUM** (40-70%): Moderate risk, alerts
- **HIGH** (70-100%): High risk, immediate action

### Real-time Monitoring
- **5-minute intervals** - Continuous solar activity monitoring
- **Automatic alerts** - Risk-based notification system
- **Connection status** - WebSocket health monitoring
- **Data refresh** - Automatic data updates

### Data Sources
- **NASA DONKI API** - Solar flare, CME, and storm data
- **Real-time ingestion** - Automated data collection
- **Historical analysis** - 7-day data retention
- **Feature engineering** - 51 comprehensive features

## 🔧 Configuration

### Environment Variables
Create `.env` files in respective directories:

**Backend (.env):**
```env
DATABASE_URL=sqlite:///flarealert.db
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

**Frontend (.env):**
```env
REACT_APP_API_URL=http://localhost:8000
```

### Alert Configuration
- **Risk Threshold**: Configurable probability threshold
- **Email Notifications**: SMTP integration
- **Webhook URLs**: Custom notification endpoints
- **Alert Frequency**: Real-time or batched

## 📈 Performance Metrics

### Model Performance
- **AUC**: 0.772
- **Precision**: 0.502
- **Recall**: 0.176
- **False Positive Rate**: 0.048

### System Performance
- **Prediction Latency**: < 100ms
- **WebSocket Latency**: < 50ms
- **Data Refresh**: 1-5 minutes
- **Uptime**: 99.9% target

## 🔒 Security Features

- **CORS Configuration** - Cross-origin resource sharing
- **Input Validation** - Pydantic model validation
- **Error Handling** - Comprehensive error management
- **Rate Limiting** - API request throttling
- **Health Monitoring** - System status checks

## 🚀 Deployment

### Development
```bash
# Backend
cd backend && python -m uvicorn app.main:app --reload

# Frontend
cd frontend && npm start
```

### Production
```bash
# Backend
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm run build
```

### Docker (Future)
```bash
docker-compose up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NASA DONKI API** - Solar activity data
- **XGBoost** - Machine learning framework
- **FastAPI** - Web framework
- **React** - Frontend framework
- **Tailwind CSS** - Styling framework

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the API docs at `/docs`

---

**FlareAlert** - Predicting solar storms, protecting our future 🌞⚡
