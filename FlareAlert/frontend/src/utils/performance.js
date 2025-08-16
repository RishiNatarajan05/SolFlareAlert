// Performance optimization utilities

// Debounce function to limit function calls
export const debounce = (func, wait) => {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
};

// Throttle function to limit function calls
export const throttle = (func, limit) => {
  let inThrottle;
  return function() {
    const args = arguments;
    const context = this;
    if (!inThrottle) {
      func.apply(context, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};

// Check if user prefers reduced motion
export const prefersReducedMotion = () => {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
};

// Optimize animations based on user preference
export const getAnimationConfig = (defaultConfig, reducedConfig = {}) => {
  if (prefersReducedMotion()) {
    return {
      duration: 0,
      delay: 0,
      ...reducedConfig
    };
  }
  return defaultConfig;
};

// Memoization helper for expensive calculations
export const memoize = (fn) => {
  const cache = new Map();
  return (...args) => {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key);
    }
    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
};

// Intersection Observer for lazy loading
export const createIntersectionObserver = (callback, options = {}) => {
  const defaultOptions = {
    root: null,
    rootMargin: '50px',
    threshold: 0.1,
    ...options
  };
  
  return new IntersectionObserver(callback, defaultOptions);
};

// Performance monitoring
export const measurePerformance = (name, fn) => {
  if (process.env.NODE_ENV === 'development') {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    console.log(`${name} took ${end - start}ms`);
    return result;
  }
  return fn();
};

// Optimize React Query settings for better performance
export const getOptimizedQueryConfig = (baseConfig = {}) => ({
  staleTime: 300000, // 5 minutes
  cacheTime: 600000, // 10 minutes
  refetchOnWindowFocus: false,
  refetchOnMount: false,
  retry: 1,
  ...baseConfig
});

// WebSocket optimization settings
export const getWebSocketConfig = () => ({
  throttleInterval: 1000, // 1 second
  maxReconnectAttempts: 5,
  reconnectDelay: 3000,
  heartbeatInterval: 30000, // 30 seconds
});

// Animation performance settings
export const getAnimationSettings = () => ({
  duration: prefersReducedMotion() ? 0 : 300,
  ease: 'easeOut',
  delay: prefersReducedMotion() ? 0 : 100,
});

// Memory management
export const cleanupResources = () => {
  // Clear any cached data
  if (window.caches) {
    caches.keys().then(names => {
      names.forEach(name => {
        caches.delete(name);
      });
    });
  }
  
  // Clear any stored data if needed
  // localStorage.clear(); // Uncomment if you want to clear localStorage
};

// Export default configuration
export default {
  debounce,
  throttle,
  prefersReducedMotion,
  getAnimationConfig,
  memoize,
  createIntersectionObserver,
  measurePerformance,
  getOptimizedQueryConfig,
  getWebSocketConfig,
  getAnimationSettings,
  cleanupResources,
};
