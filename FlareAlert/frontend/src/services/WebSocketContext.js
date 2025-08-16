import React, { createContext, useContext, useEffect, useState } from 'react';
import toast from 'react-hot-toast';

const WebSocketContext = createContext();

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [latestData, setLatestData] = useState(null);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        toast.success('Real-time connection established');
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLatestData(data);
          
          // Handle different message types
          switch (data.type) {
            case 'solar_flare_alert':
              handleAlert(data);
              break;
            case 'status_update':
              // Update real-time status
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
        
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
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
  }, []);

  const handleAlert = (alertData) => {
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
    
    toast(alertData.message, {
      icon: '⚠️',
      style: {
        background: riskColors[alertData.risk_level] === 'red' ? '#dc2626' : '#1f2937',
        color: '#fff',
      },
      duration: 8000,
    });
  };

  const sendMessage = (message) => {
    if (socket && isConnected) {
      socket.send(JSON.stringify(message));
    }
  };

  const clearAlerts = () => {
    setAlerts([]);
  };

  const value = {
    isConnected,
    latestData,
    alerts,
    sendMessage,
    clearAlerts,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};
