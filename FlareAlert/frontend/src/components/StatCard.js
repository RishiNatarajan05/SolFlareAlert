import React, { memo } from 'react';
import { motion } from 'framer-motion';

const StatCard = memo(({ title, value, icon: Icon, color, subtitle, delay = 0 }) => {
  // Check for reduced motion preference
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  return (
    <motion.div
      initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: 20 }}
      animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
      transition={{ 
        duration: prefersReducedMotion ? 0 : 0.3,
        delay: prefersReducedMotion ? 0 : delay * 0.1 // Reduced delay multiplier
      }}
      whileHover={prefersReducedMotion ? {} : { 
        y: -2,
        transition: { duration: 0.15 } // Reduced from 0.2
      }}
      className="card hover:shadow-lg transition-shadow duration-200"
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-400 mb-1">{title}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
          {subtitle && (
            <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
          )}
        </div>
        <div className={`p-3 rounded-lg bg-gray-700 ${color}`}>
          <Icon className="w-6 h-6" />
        </div>
      </div>
    </motion.div>
  );
});

StatCard.displayName = 'StatCard';

export default StatCard;
