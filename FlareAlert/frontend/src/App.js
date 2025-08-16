import React, { memo } from 'react';
import { Routes, Route } from 'react-router-dom';
import { motion } from 'framer-motion';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Predictions from './pages/Predictions';
import SolarActivity from './pages/SolarActivity';
import ModelInfo from './pages/ModelInfo';
import Settings from './pages/Settings';
import { WebSocketProvider } from './services/WebSocketContext';

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const App = memo(() => {
  return (
    <WebSocketProvider>
      <div className="flex h-screen bg-gray-900">
        <Sidebar />
        <main className="flex-1 overflow-auto">
          <motion.div
            initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: 10 }}
            animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
            transition={{ duration: prefersReducedMotion ? 0 : 0.3 }}
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
});

App.displayName = 'App';

export default App;
