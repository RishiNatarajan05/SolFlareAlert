import React from 'react';
import { motion } from 'framer-motion';

const RiskGauge = ({ value, riskLevel, size = 'md' }) => {
  const sizeClasses = {
    sm: 'w-24 h-24',
    md: 'w-32 h-32',
    lg: 'w-48 h-48'
  };

  const textSizes = {
    sm: 'text-lg',
    md: 'text-2xl',
    lg: 'text-4xl'
  };

  const strokeWidths = {
    sm: 4,
    md: 6,
    lg: 8
  };

  const radius = size === 'sm' ? 40 : size === 'md' ? 56 : 88;
  const circumference = 2 * Math.PI * radius;
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (value / 100) * circumference;

  const getRiskColor = (level) => {
    switch (level) {
      case 'MINIMAL':
        return '#4ade80';
      case 'LOW':
        return '#fbbf24';
      case 'MEDIUM':
        return '#ff8c42';
      case 'HIGH':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const getRiskLabel = (level) => {
    switch (level) {
      case 'MINIMAL':
        return 'Safe';
      case 'LOW':
        return 'Low Risk';
      case 'MEDIUM':
        return 'Medium Risk';
      case 'HIGH':
        return 'High Risk';
      default:
        return 'Unknown';
    }
  };

  return (
    <div className={`relative ${sizeClasses[size]}`}>
      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
        {/* Background circle */}
        <circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke="#374151"
          strokeWidth={strokeWidths[size]}
        />
        
        {/* Progress circle */}
        <motion.circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke={getRiskColor(riskLevel)}
          strokeWidth={strokeWidths[size]}
          strokeLinecap="round"
          strokeDasharray={strokeDasharray}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      </svg>
      
      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className={`font-bold ${textSizes[size]} ${riskLevel === 'HIGH' ? 'animate-pulse' : ''}`}
          style={{ color: getRiskColor(riskLevel) }}
        >
          {Math.round(value * 100)}%
        </motion.div>
        <div className="text-xs text-gray-400 mt-1">
          {getRiskLabel(riskLevel)}
        </div>
      </div>
      
      {/* Risk level indicator */}
      <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2">
        <div 
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: getRiskColor(riskLevel) }}
        />
      </div>
    </div>
  );
};

export default RiskGauge;
