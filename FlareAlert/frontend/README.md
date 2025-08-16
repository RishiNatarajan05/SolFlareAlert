# FlareAlert Frontend

A modern React dashboard for real-time solar flare prediction and monitoring.

## Features

- 🎯 **Real-time Dashboard** - Live solar flare risk monitoring with animated gauges
- 📊 **Interactive Charts** - Historical data visualization and trends
- 🔔 **Real-time Alerts** - WebSocket-powered notifications for high-risk events
- 📱 **Responsive Design** - Works on desktop, tablet, and mobile devices
- 🎨 **Modern UI** - Dark theme with smooth animations and transitions
- ⚡ **Fast Performance** - Optimized with React Query for efficient data fetching

## Tech Stack

- **React 18** - Modern React with hooks and functional components
- **React Router** - Client-side routing
- **React Query** - Server state management and caching
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Smooth animations and transitions
- **Lucide React** - Beautiful icons
- **Axios** - HTTP client for API communication
- **WebSocket** - Real-time communication with backend

## Getting Started

### Prerequisites

- Node.js 16+ 
- npm or yarn
- FlareAlert backend running on port 8000

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm start
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
npm run build
```

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Sidebar.js      # Navigation sidebar
│   └── RiskGauge.js    # Animated risk visualization
├── pages/              # Page components
│   ├── Dashboard.js    # Main dashboard
│   ├── Predictions.js  # Prediction analysis
│   ├── SolarActivity.js # Solar activity monitoring
│   ├── ModelInfo.js    # Model information
│   └── Settings.js     # System settings
├── services/           # API and WebSocket services
│   ├── api.js         # REST API client
│   └── WebSocketContext.js # Real-time updates
├── styles/            # Global styles
│   └── index.css      # Tailwind and custom styles
├── App.js             # Main app component
└── index.js           # Entry point
```

## Features Overview

### Dashboard
- Real-time risk gauge with color-coded levels
- Current solar activity statistics
- Recent alerts and notifications
- Model performance metrics

### Predictions
- Interactive date selection for predictions
- Detailed prediction analysis
- Risk level explanations
- Historical prediction data

### Solar Activity
- Real-time solar flare counts
- CME (Coronal Mass Ejection) monitoring
- Geomagnetic storm tracking
- Historical activity charts

### Model Information
- Model performance metrics (AUC, Precision, Recall)
- Feature count and model details
- Last update timestamps
- Model configuration information

### Settings
- Alert threshold configuration
- Email and webhook notifications
- Data ingestion controls
- Model retraining triggers

## API Integration

The frontend communicates with the FlareAlert backend API:

- **Base URL**: `http://localhost:8000`
- **WebSocket**: `ws://localhost:8000/ws`
- **Endpoints**: See backend documentation for available endpoints

## Real-time Features

- **WebSocket Connection** - Automatic reconnection on disconnection
- **Live Updates** - Real-time prediction and activity updates
- **Alert Notifications** - Toast notifications for high-risk events
- **Status Monitoring** - Connection status indicator in sidebar

## Styling

The app uses a custom dark theme with solar-inspired colors:

- **Primary**: Blue tones for main UI elements
- **Solar Flare**: Orange/red for flare-related data
- **CME**: Orange for coronal mass ejection data
- **Storm**: Yellow for geomagnetic storm data
- **Safe**: Green for low-risk indicators

## Development

### Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm eject` - Eject from Create React App

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is part of the FlareAlert system. See the main project license for details.
