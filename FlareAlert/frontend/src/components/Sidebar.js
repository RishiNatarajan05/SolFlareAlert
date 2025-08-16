import React from 'react';
import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Home, 
  TrendingUp, 
  Activity, 
  Database, 
  Settings,
  Wifi,
  WifiOff
} from 'lucide-react';
import { useWebSocket } from '../services/WebSocketContext';

const navItems = [
  { path: '/', icon: Home, label: 'Dashboard' },
  { path: '/predictions', icon: TrendingUp, label: 'Predictions' },
  { path: '/solar-activity', icon: Activity, label: 'Solar Activity' },
  { path: '/model-info', icon: Database, label: 'Model Info' },
  { path: '/settings', icon: Settings, label: 'Settings' },
];

const Sidebar = () => {
  const { isConnected } = useWebSocket();

  return (
    <motion.div
      initial={{ x: -250 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.5 }}
      className="w-64 bg-gray-800 border-r border-gray-700 flex flex-col"
    >
      {/* Header */}
      <div className="p-6 border-b border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-r from-orange-500 to-red-500 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">F</span>
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">FlareAlert</h1>
            <p className="text-sm text-gray-400">Solar Monitoring</p>
          </div>
        </div>
        
        {/* Connection Status */}
        <div className="mt-4 flex items-center space-x-2">
          {isConnected ? (
            <Wifi className="w-4 h-4 text-green-400" />
          ) : (
            <WifiOff className="w-4 h-4 text-red-400" />
          )}
          <span className={`text-sm ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors duration-200 ${
                    isActive
                      ? 'bg-primary-600 text-white'
                      : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                  }`
                }
              >
                <item.icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700">
        <div className="text-center">
          <p className="text-xs text-gray-500">FlareAlert v1.0.0</p>
          <p className="text-xs text-gray-500 mt-1">Real-time Solar Monitoring</p>
        </div>
      </div>
    </motion.div>
  );
};

export default Sidebar;
