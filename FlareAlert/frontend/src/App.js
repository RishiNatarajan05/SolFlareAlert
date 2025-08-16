import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { motion } from 'framer-motion';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Predictions from './pages/Predictions';
import SolarActivity from './pages/SolarActivity';
import ModelInfo from './pages/ModelInfo';
import Settings from './pages/Settings';
import { WebSocketProvider } from './services/WebSocketContext';

function App() {
  return (
    <WebSocketProvider>
      <div className="flex h-screen bg-gray-900">
        <Sidebar />
        <main className="flex-1 overflow-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="p-6"
          >
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/solar-activity" element={<SolarActivity />} />
              <Route path="/model-info" element={<ModelInfo />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </motion.div>
        </main>
      </div>
    </WebSocketProvider>
  );
}

export default App;
