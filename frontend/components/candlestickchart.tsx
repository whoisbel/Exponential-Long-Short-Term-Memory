"use client";
import React, { useMemo } from "react";
import dynamic from "next/dynamic";
import { OHLCType } from "@/types";
import { ApexOptions } from "apexcharts";

// Dynamically import ApexCharts with SSR disabled for Next.js
const ApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

interface LastWeekDataItem {
  date: string;
  elu: number;
  actual?: number | null;
}

const CandlestickChart = ({
  dates,
  ohlc,
  height = 700,
  last_week_data,
}: {
  dates: string[];
  ohlc: OHLCType[];
  height?: number;
  last_week_data?: LastWeekDataItem[];
}) => {
  // Process data for candlestick chart only
  const chartData = useMemo(() => {
    try {
      // Validate inputs
      if (!ohlc || !Array.isArray(ohlc) || ohlc.length === 0) {
        return { isValid: false, error: 'No OHLC data' };
      }
      
      if (!dates || !Array.isArray(dates) || dates.length === 0) {
        return { isValid: false, error: 'No dates data' };
      }

      // Process candlestick data with extra validation
      const candlestickData = [];
      for (let i = 0; i < Math.min(ohlc.length, dates.length); i++) {
        const item = ohlc[i];
        const date = dates[i];
        
        if (!item || !date) {
          continue;
        }
        
        // Validate all OHLC values exist and are numbers
        if (typeof item.open !== 'number' || typeof item.high !== 'number' || 
            typeof item.low !== 'number' || typeof item.close !== 'number') {
          continue;
        }

        // Additional validation: check for NaN or infinite values
        if (isNaN(item.open) || isNaN(item.high) || isNaN(item.low) || isNaN(item.close) ||
            !isFinite(item.open) || !isFinite(item.high) || !isFinite(item.low) || !isFinite(item.close)) {
          continue;
        }

        // Validate date
        const timestamp = new Date(date).getTime();
        if (isNaN(timestamp)) {
          continue;
        }

        const candlestickPoint = {
          x: timestamp,
          y: [
            Number(item.open.toFixed(2)),
            Number(item.high.toFixed(2)),
            Number(item.low.toFixed(2)),
            Number(item.close.toFixed(2)),
          ],
        };

        // Final validation of the y array
        if (candlestickPoint.y.length !== 4 || candlestickPoint.y.some(val => isNaN(val) || !isFinite(val))) {
          continue;
        }

        candlestickData.push(candlestickPoint);
      }

      // Process prediction data
      let predictionData = [];
      if (last_week_data && Array.isArray(last_week_data) && last_week_data.length > 0) {
        // Use all available prediction data instead of limiting to 10
        for (const item of last_week_data) {
          if (!item || !item.date || typeof item.elu !== 'number' || isNaN(item.elu) || !isFinite(item.elu)) {
            continue;
          }
          
          const timestamp = new Date(item.date).getTime();
          if (isNaN(timestamp)) continue;

          predictionData.push({
            x: timestamp,
            y: Number(item.elu.toFixed(2)),
          });
        }
      }

      return {
        isValid: candlestickData.length > 0,
        candlestickData,
        predictionData,
      };
    } catch (error) {
      console.error('Error processing candlestick chart data:', error);
      return { isValid: false, error: 'Data processing error' };
    }
  }, [ohlc, dates, last_week_data]);

  // Early return if data is invalid
  if (!chartData.isValid) {
    return (
      <div className="flex justify-center items-center h-full">
        <p className="text-gray-500">Loading candlestick chart data...</p>
      </div>
    );
  }

  const { candlestickData, predictionData } = chartData;

  // Candlestick chart specific options
  const options: ApexOptions = {
    chart: {
      type: "candlestick",
      height: height || 400,
      animations: {
        enabled: true,
        
        speed: 800
      },
      toolbar: {
        show: true
      }
    },
    title: {
      text: "OHLC Candlestick Chart",
      align: "left",
      style: {
        fontSize: '16px',
        fontWeight: 600
      }
    },
    xaxis: {
      type: "datetime",
      labels: {
        datetimeUTC: false,
        format: 'MMM dd'
      }
    },
    yaxis: {
      title: {
        text: 'Price ($)'
      },
      labels: {
        formatter: function(val) {
          return val ? `$${Number(val).toFixed(2)}` : '$0.00';
        }
      }
    },
    legend: {
      show: true,
      position: 'top'
    },
    stroke: {
      width: [1, 6], // Thin candlestick, thicker prediction
      curve: ['straight', 'smooth'] // Different curves for each series
    },
    markers: {
      size: [0, 10], // No markers for candlestick, large markers for prediction
      colors: ['transparent', '#FF4560'],
      strokeColors: ['transparent', '#fff'],
      strokeWidth: [0, 4],
      
    },
    plotOptions: {
      candlestick: {
        colors: {
          upward: '#00B746',
          downward: '#EF403C'
        },
        wick: {
          useFillColor: true
        }
      }
    },
    tooltip: {
      shared: false,
      intersect: true,
      y: {
        formatter: function(val) {
          return val ? `$${Number(val).toFixed(2)}` : '$0.00';
        }
      }
    },
    grid: {
      borderColor: '#e7e7e7',
      strokeDashArray: 3
    }
  };

  // Build series for candlestick chart
  const series = [];

  // Add candlestick data
  if (candlestickData && candlestickData.length > 0) {
    series.push({
      name: 'OHLC',
      type: 'candlestick' as const,
      data: candlestickData,
    });
  }

  // Add prediction line overlay
  if (predictionData && predictionData.length > 0) {
    series.push({
      name: 'ELU Prediction',
      type: 'line' as const,
      data: predictionData,
      color: '#FF4560',
      stroke: {
        width: 6,
        curve: 'smooth' as const,
      },
    });
  }

  return (
    <div className="w-full h-full" style={{ minHeight: `${height}px` }}>
      <ApexChart
        options={options}
        series={series as any}
        type="candlestick"
        height="100%"
      />
    </div>
  );
};

export default React.memo(CandlestickChart);