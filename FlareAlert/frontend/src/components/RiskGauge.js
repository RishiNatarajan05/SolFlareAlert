import React from 'react';
import { motion } from 'framer-motion';

const RiskGauge = ({ value = 0, riskLevel, size = 'md' }) => {
  const sizeClasses = {
    sm: 'w-24 h-24',
    md: 'w-32 h-32',
    lg: 'w-48 h-48'
  };

  const textSizes = {
    sm: 'text-xl',
    md: 'text-3xl',
    lg: 'text-5xl'
  };

  const strokeWidths = {
    sm: 4,
    md: 6,
    lg: 8
  };

  // Clamp value to [0,1]
  const v = Math.max(0, Math.min(1, value));

  const getRiskColor = (level) => {
    switch (level) {
      case 'MINIMAL': return '#4ade80';
      case 'LOW':     return '#fbbf24';
      case 'MEDIUM':  return '#ff8c42';
      case 'HIGH':    return '#ef4444';
      default:        return '#6b7280';
    }
  };

  const getRiskLabel = (level) => {
    switch (level) {
      case 'MINIMAL': return 'Safe';
      case 'LOW':     return 'Low Risk';
      case 'MEDIUM':  return 'Medium Risk';
      case 'HIGH':    return 'High Risk';
      default:        return 'Unknown';
    }
  };

  const stroke = strokeWidths[size];
  // Keep radius inside 120x120 viewBox (half is 60). Leave a little padding.
  const radius = 60 - stroke / 2 - 4;
  const circumference = 2 * Math.PI * radius;
  const targetOffset = circumference * (1 - v);

  return (
    <motion.div 
      className={`relative ${sizeClasses[size]} bg-gradient-to-br from-gray-900 to-gray-800 rounded-full p-2 ${riskLevel === 'HIGH' ? 'animate-pulse' : ''}`}
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      role="img"
      aria-label={`Risk ${Math.round(v * 100)} percent: ${getRiskLabel(riskLevel)}`}
    >
      <svg className="w-full h-full" viewBox="0 0 120 120">
        {/* Track */}
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke="#374151"
          strokeWidth={stroke}
          opacity="0.5"
        />
        
        {/* Progress arc */}
        <motion.circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke={getRiskColor(riskLevel)}
          strokeWidth={stroke}
          strokeLinecap="round"
          // Show % of circle: full dasharray, offset is what's left
          strokeDasharray={`${circumference} ${circumference}`}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: targetOffset }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          // Start at 12 o'clock
          transform="rotate(-90 60 60)"
          style={{ filter: 'drop-shadow(0 0 8px currentColor)' }}
        />
      </svg>
      
      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.4, duration: 0.4 }}
          className={`font-bold ${textSizes[size]} drop-shadow-lg`}
          style={{ color: getRiskColor(riskLevel) }}
        >
          {Math.round(v * 100)}%
        </motion.div>
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.4 }}
          className="text-sm text-gray-300 mt-1 font-medium"
        >
          {getRiskLabel(riskLevel)}
        </motion.div>
      </div>
      
      {/* Risk level indicator */}
      <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2">
        <motion.div 
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.9, duration: 0.25 }}
          className="w-4 h-4 rounded-full shadow-lg"
          style={{ 
            backgroundColor: getRiskColor(riskLevel),
            boxShadow: `0 0 12px ${getRiskColor(riskLevel)}40`
          }}
        />
      </div>
    </motion.div>
  );
};

export default RiskGauge;
