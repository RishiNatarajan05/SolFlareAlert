import React, { useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const SolarActivityChart = ({ data, isLoading }) => {
  const chartRef = useRef(null);
  
  const [visibleSeries, setVisibleSeries] = useState({
    flares: true,
    cmes: true,
    storms: true,
    kp: true
  });
  
  const prefersReducedMotion = useMemo(() => 
    window.matchMedia('(prefers-reduced-motion: reduce)').matches, 
    []
  );

  // Process data for chart
  const chartData = useMemo(() => {
    if (!data) {
      // Return sample data for testing
      return {
        labels: ['7/15', '7/16', '7/17', '7/18', '7/19', '7/20', '7/21', '7/22', '7/23', '7/24', '7/25', '7/26', '7/27', '7/28', '7/29', '7/30', '7/31', '8/1', '8/2', '8/3', '8/4', '8/5', '8/6', '8/7', '8/8', '8/9', '8/10', '8/11', '8/12', '8/13'],
        datasets: [
          {
            label: 'Solar Flares',
            data: [2, 1, 3, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 0, 0, 2, 1, 2, 0, 3, 2, 2, 2, 2, 2, 2, 4, 3, 1, 1],
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 3,
            pointHoverRadius: 5,
            hidden: !visibleSeries.flares,
          },
          {
            label: 'CMEs',
            data: [3, 2, 4, 1, 3, 2, 1, 5, 2, 2, 1, 2, 1, 1, 2, 4, 3, 2, 2, 4, 3, 2, 3, 4, 4, 3, 3, 2, 2, 1],
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 3,
            pointHoverRadius: 5,
            hidden: !visibleSeries.cmes,
          },
          {
            label: 'Geomagnetic Storms',
            data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 3,
            pointHoverRadius: 5,
            hidden: !visibleSeries.storms,
          },
          {
            label: 'Kp Index (Avg)',
            data: [2.1, 2.3, 2.0, 2.5, 2.2, 2.4, 2.1, 2.3, 2.0, 2.2, 2.1, 2.3, 2.0, 2.2, 2.1, 2.4, 2.2, 2.1, 2.3, 2.0, 2.2, 2.1, 2.3, 2.0, 2.2, 6.0, 2.1, 2.3, 2.0, 2.2],
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.4,
            pointRadius: 3,
            pointHoverRadius: 5,
            yAxisID: 'y1',
            hidden: !visibleSeries.kp,
          },
        ].filter(dataset => !dataset.hidden),
      };
    }

    // Group data by day
    const dailyData = {};
    
    // Process flares
    data.flares?.forEach(flare => {
      const date = new Date(flare.time).toLocaleDateString();
      if (!dailyData[date]) {
        dailyData[date] = { flares: 0, cmes: 0, storms: 0, kp: [] };
      }
      dailyData[date].flares += 1;
    });

    // Process CMEs
    data.cmes?.forEach(cme => {
      const date = new Date(cme.time).toLocaleDateString();
      if (!dailyData[date]) {
        dailyData[date] = { flares: 0, cmes: 0, storms: 0, kp: [] };
      }
      dailyData[date].cmes += 1;
    });

    // Process storms and Kp index
    data.storms?.forEach(storm => {
      const date = new Date(storm.time).toLocaleDateString();
      if (!dailyData[date]) {
        dailyData[date] = { flares: 0, cmes: 0, storms: 0, kp: [] };
      }
      dailyData[date].storms += 1;
      dailyData[date].kp.push(storm.kp);
    });



    console.log('SolarActivityChart - Processed daily data:', dailyData);

    // Sort by date and get last 30 days
    const sortedDates = Object.keys(dailyData).sort((a, b) => new Date(a) - new Date(b));
    
    // Use the actual dates from the data, but limit to last 30
    const last30Days = sortedDates.slice(-30);



    const allDatasets = [
      {
        label: 'Solar Flares',
        data: last30Days.map(date => dailyData[date]?.flares || 0),
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 5,
        hidden: !visibleSeries.flares,
      },
      {
        label: 'CMEs',
        data: last30Days.map(date => dailyData[date]?.cmes || 0),
        borderColor: '#ef4444',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 5,
        hidden: !visibleSeries.cmes,
      },
      {
        label: 'Geomagnetic Storms',
        data: last30Days.map(date => dailyData[date]?.storms || 0),
        borderColor: '#8b5cf6',
        backgroundColor: 'rgba(139, 92, 246, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 5,
        hidden: !visibleSeries.storms,
      },
      {
        label: 'Kp Index (Avg)',
        data: last30Days.map(date => {
          const dayData = dailyData[date];
          if (!dayData || !dayData.kp || dayData.kp.length === 0) {
            return 0;
          }
          return dayData.kp.reduce((sum, kp) => sum + kp, 0) / dayData.kp.length;
        }),
        borderColor: '#06b6d4',
        backgroundColor: 'rgba(6, 182, 212, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.4,
        pointRadius: 3,
        pointHoverRadius: 5,
        yAxisID: 'y1',
        hidden: !visibleSeries.kp,
      },
    ];

    const result = {
      labels: last30Days.map(date => {
        const d = new Date(date);
        return `${d.getMonth() + 1}/${d.getDate()}`;
      }),
      datasets: allDatasets.filter(dataset => !dataset.hidden),
    };
    
    return result;
  }, [data, visibleSeries]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#e5e7eb',
          usePointStyle: true,
          padding: 20,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(17, 24, 39, 0.9)',
        titleColor: '#e5e7eb',
        bodyColor: '#e5e7eb',
        borderColor: '#374151',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(75, 85, 99, 0.2)',
        },
        ticks: {
          color: '#9ca3af',
          maxRotation: 45,
        },
      },
      y: {
        type: 'linear',
        display: true,
        position: 'left',
        grid: {
          color: 'rgba(75, 85, 99, 0.2)',
        },
        ticks: {
          color: '#9ca3af',
        },
        title: {
          display: true,
          text: 'Count',
          color: '#9ca3af',
        },
      },
      y1: {
        type: 'linear',
        display: true,
        position: 'right',
        grid: {
          drawOnChartArea: false,
        },
        ticks: {
          color: '#9ca3af',
        },
        title: {
          display: true,
          text: 'Kp Index',
          color: '#9ca3af',
        },
      },
    },
  }), []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="loading-spinner"></div>
      </div>
    );
  }

  if (!chartData || !data) {
    return (
      <div className="text-center text-gray-400 py-8">
        <div className="w-12 h-12 mx-auto mb-4 text-gray-600">
          <svg fill="currentColor" viewBox="0 0 20 20">
            <path d="M2 10a8 8 0 018-8v8h8a8 8 0 11-16 0z" />
            <path d="M12 2.252A8.014 8.014 0 0117.748 8H12V2.252z" />
          </svg>
        </div>
        <p>No historical solar activity data available</p>
        <p className="text-sm mt-2">Data will appear here once solar activity is recorded</p>
      </div>
    );
  }

  return (
    <motion.div
      initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, y: 20 }}
      animate={prefersReducedMotion ? { opacity: 1 } : { opacity: 1, y: 0 }}
      transition={{ duration: prefersReducedMotion ? 0 : 0.5 }}
      className="w-full"
    >
      <div className="mb-4">
        <h3 className="text-lg font-medium text-white mb-2">Solar Activity Trends (30 Days)</h3>
        <p className="text-sm text-gray-400">
          Daily counts of solar flares, CMEs, geomagnetic storms, and average Kp index
        </p>
      </div>
      
      {/* Interactive Legend with Checkboxes */}
      <div className="mb-4 p-3 bg-gray-800 rounded-lg border border-gray-700">
        <div className="flex justify-between items-center mb-3">
          <h4 className="text-sm font-medium text-white">Toggle Data Series:</h4>
          <div className="flex space-x-2">
            <button
              onClick={() => setVisibleSeries({ flares: true, cmes: true, storms: true, kp: true })}
              className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
            >
              Select All
            </button>
            <button
              onClick={() => setVisibleSeries({ flares: false, cmes: false, storms: false, kp: false })}
              className="text-xs px-2 py-1 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
            >
              Clear All
            </button>
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={visibleSeries.flares}
              onChange={(e) => setVisibleSeries(prev => ({ ...prev, flares: e.target.checked }))}
              className="w-4 h-4 text-yellow-500 bg-gray-700 border-gray-600 rounded focus:ring-yellow-500 focus:ring-2"
            />
            <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
            <span className="text-sm text-gray-300">Solar Flares</span>
          </label>
          
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={visibleSeries.cmes}
              onChange={(e) => setVisibleSeries(prev => ({ ...prev, cmes: e.target.checked }))}
              className="w-4 h-4 text-red-500 bg-gray-700 border-gray-600 rounded focus:ring-red-500 focus:ring-2"
            />
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span className="text-sm text-gray-300">CMEs</span>
          </label>
          
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={visibleSeries.storms}
              onChange={(e) => setVisibleSeries(prev => ({ ...prev, storms: e.target.checked }))}
              className="w-4 h-4 text-purple-500 bg-gray-700 border-gray-600 rounded focus:ring-purple-500 focus:ring-2"
            />
            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
            <span className="text-sm text-gray-300">Geomagnetic Storms</span>
          </label>
          
          <label className="flex items-center space-x-2 cursor-pointer">
            <input
              type="checkbox"
              checked={visibleSeries.kp}
              onChange={(e) => setVisibleSeries(prev => ({ ...prev, kp: e.target.checked }))}
              className="w-4 h-4 text-cyan-500 bg-gray-700 border-gray-600 rounded focus:ring-cyan-500 focus:ring-2"
            />
            <div className="w-3 h-3 bg-cyan-500 rounded-full"></div>
            <span className="text-sm text-gray-300">Kp Index</span>
          </label>
        </div>
      </div>
      
      <div className="h-80 w-full">
        <Line ref={chartRef} data={chartData} options={chartOptions} />
      </div>
    </motion.div>
  );
};

export default SolarActivityChart;
