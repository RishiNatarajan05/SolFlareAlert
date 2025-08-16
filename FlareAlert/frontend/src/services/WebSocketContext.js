import React, { createContext, useContext, useEffect, useState, useCallback, useMemo } from 'react';
import toast from 'react-hot-toast';
import config from './config';

const WebSocketContext = createContext();

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

// Throttle function to limit update frequency
const throttle = (func, limit) => {
  let inThrottle;
  return function() {
    const args = arguments;
    const context = this;
    if (!inThrottle) {
      func.apply(context, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  }
};

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [latestData, setLatestData] = useState(null);
  const [alerts, setAlerts] = useState([]);

  // Throttled data update function
  const throttledSetLatestData = useCallback(
    throttle((data) => {
      setLatestData(data);
    }, config.WEBSOCKET.THROTTLE_INTERVAL),
    []
  );

  useEffect(() => {
    const connectWebSocket = () => {
      const wsUrl = config.WS_BASE_URL + '/ws';
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        toast.success('Real-time connection established');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          // Use throttled update for status updates to prevent excessive re-renders
          if (data.type === 'status_update') {
            throttledSetLatestData(data);
          } else {
            setLatestData(data);
          }
          
          // Handle different message types
          switch (data.type) {
            case 'solar_flare_alert':
              handleAlert(data);
              break;
            case 'data_ingestion_complete':
              toast.success('Data ingestion completed');
              break;
            case 'model_retraining_complete':
              toast.success('Model retraining completed');
              break;
            default:
              break;
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        toast.error('Real-time connection lost');
        
        // Attempt to reconnect after configurable delay
        setTimeout(connectWebSocket, config.WEBSOCKET.RECONNECT_DELAY);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };
      
      setSocket(ws);
    };
    
    connectWebSocket();
    
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [throttledSetLatestData]);

  const handleAlert = useCallback((alertData) => {
    const newAlert = {
      id: Date.now(),
      ...alertData,
      timestamp: new Date().toISOString(),
    };
    
    setAlerts(prev => [newAlert, ...prev.slice(0, 9)]); // Keep last 10 alerts
    
    // Show toast notification
    const riskColors = {
      'MINIMAL': 'green',
      'LOW': 'yellow',
      'MEDIUM': 'orange',
      'HIGH': 'red'
    };
    
    // Use configurable duration based on risk level
    const duration = alertData.risk_level === 'HIGH' 
      ? config.ALERTS.HIGH_RISK_DURATION 
      : config.ALERTS.TOAST_DURATION;
    
    toast(alertData.message, {
      icon: '⚠️',
      style: {
        background: riskColors[alertData.risk_level] === 'red' ? '#dc2626' : '#1f2937',
        color: '#fff',
      },
      duration: duration,
    });
  }, []);

  const sendMessage = useCallback((message) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify(message));
    }
  }, [socket, isConnected]);

  const clearAlerts = useCallback(() => {
    setAlerts([]);
  }, []);

  // Memoize the context value to prevent unnecessary re-renders
  const value = useMemo(() => ({
    isConnected,
    latestData,
    alerts,
    sendMessage,
    clearAlerts,
  }), [isConnected, latestData, alerts, sendMessage, clearAlerts]);

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};
