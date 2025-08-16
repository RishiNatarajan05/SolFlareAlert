import React, { useEffect, useRef, memo } from 'react';
import { motion } from 'framer-motion';
import { useSpring, animated, config } from '@react-spring/web';

const RiskGauge = memo(({ value = 0, riskLevel, size = 'md' }) => {
  const gaugeRef = useRef(null);
  
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
      case 'LOW': return '#fbbf24';
      case 'MEDIUM': return '#ff8c42';
      case 'HIGH': return '#ef4444';
      default: return '#6b7280';
    }
  };

  const getRiskLabel = (level) => {
    switch (level) {
      case 'MINIMAL': return 'Safe';
      case 'LOW': return 'Low Risk';
      case 'MEDIUM': return 'Medium Risk';
      case 'HIGH': return 'High Risk';
      default: return 'Unknown';
    }
  };

  const stroke = strokeWidths[size];
  const radius = 60 - stroke / 2 - 4;
  const circumference = 2 * Math.PI * radius;
  const targetOffset = circumference * (1 - v);

  // React Spring for smooth value transitions - optimized config
  const { strokeDashoffset } = useSpring({
    strokeDashoffset: targetOffset,
    config: { tension: 120, friction: 14 }, // Faster, more responsive
    delay: 100 // Reduced delay
  });

  // Check for reduced motion preference
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  return (
    <motion.div 
      ref={gaugeRef}
      className={`relative ${sizeClasses[size]} bg-gradient-to-br from-gray-900 to-gray-800 rounded-full p-2`}
      initial={prefersReducedMotion ? { opacity: 1 } : { scale: 0, opacity: 0 }}
      animate={prefersReducedMotion ? { opacity: 1 } : { scale: 1, opacity: 1 }}
      transition={{ 
        duration: prefersReducedMotion ? 0 : 0.4, // Reduced from 0.8
        ease: "easeOut"
      }}
      whileHover={prefersReducedMotion ? {} : { 
        scale: 1.02, // Reduced from 1.05
        transition: { duration: 0.15 } // Reduced from 0.2
      }}
      role="img"
      aria-label={`Risk ${Math.round(v * 100)} percent: ${getRiskLabel(riskLevel)}`}
    >
      {/* Simplified glow effect - only for high risk */}
      {riskLevel === 'HIGH' && !prefersReducedMotion && (
        <div 
          className="absolute inset-0 rounded-full opacity-10"
          style={{ 
            background: `radial-gradient(circle, ${getRiskColor(riskLevel)} 0%, transparent 70%)`,
            filter: 'blur(4px)' // Reduced blur
          }}
        />
      )}
      
      <svg className="w-full h-full relative z-10" viewBox="0 0 120 120">
        {/* Track */}
        <animated.circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke="#374151"
          strokeWidth={stroke}
          opacity={0.5}
        />
        
        {/* Progress arc with React Spring */}
        <animated.circle
          cx="60"
          cy="60"
          r={radius}
          fill="none"
          stroke={getRiskColor(riskLevel)}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={strokeDashoffset}
          transform="rotate(-90 60 60)"
          style={{ 
            filter: riskLevel === 'HIGH' ? `drop-shadow(0 0 4px ${getRiskColor(riskLevel)})` : 'none' // Reduced glow
          }}
        />
      </svg>
      
      {/* Center content with simplified animations */}
      <div className="absolute inset-0 flex flex-col items-center justify-center z-20">
        <motion.div
          initial={prefersReducedMotion ? { opacity: 1 } : { scale: 0, y: 10 }}
          animate={prefersReducedMotion ? { opacity: 1 } : { scale: 1, y: 0 }}
          transition={{ 
            delay: prefersReducedMotion ? 0 : 0.2, // Reduced from 0.4
            duration: prefersReducedMotion ? 0 : 0.3, // Reduced from 0.6
            ease: "easeOut"
          }}
          className={`font-bold ${textSizes[size]} drop-shadow-lg`}
          style={{ color: getRiskColor(riskLevel) }}
        >
          {Math.round(v * 100)}%
        </motion.div>
        <motion.div 
          initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: 5 }}
          animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
          transition={{ 
            delay: prefersReducedMotion ? 0 : 0.3, // Reduced from 0.6
            duration: prefersReducedMotion ? 0 : 0.3, // Reduced from 0.5
            ease: "easeOut"
          }}
          className="text-sm text-gray-300 mt-1 font-medium"
        >
          {getRiskLabel(riskLevel)}
        </motion.div>
      </div>
      
      {/* Simplified risk level indicator */}
      <div className="absolute -bottom-3 left-1/2 transform -translate-x-1/2 z-20">
        <motion.div 
          initial={prefersReducedMotion ? { scale: 1 } : { scale: 0 }}
          animate={prefersReducedMotion ? { scale: 1 } : { scale: 1 }}
          transition={{ 
            delay: prefersReducedMotion ? 0 : 0.4, // Reduced from 0.9
            duration: prefersReducedMotion ? 0 : 0.2, // Reduced from 0.4
            ease: "easeOut"
          }}
          whileHover={prefersReducedMotion ? {} : { 
            scale: 1.1, // Reduced from 1.2
            transition: { duration: 0.15 } // Reduced from 0.2
          }}
          className="w-4 h-4 rounded-full shadow-lg"
          style={{ 
            backgroundColor: getRiskColor(riskLevel),
            boxShadow: riskLevel === 'HIGH' ? `0 0 8px ${getRiskColor(riskLevel)}40` : 'none' // Reduced glow
          }}
        />
      </div>
    </motion.div>
  );
});

RiskGauge.displayName = 'RiskGauge';

export default RiskGauge;
