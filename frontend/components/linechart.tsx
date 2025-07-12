"use client";
import React, { useState, useMemo } from "react";
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

const LineChart = ({
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
  // Process data for line chart only
  const chartData = useMemo(() => {
    try {
      // Validate inputs
      if (!ohlc || !Array.isArray(ohlc) || ohlc.length === 0) {
        return { isValid: false, error: 'No OHLC data' };
      }
      
      if (!dates || !Array.isArray(dates) || dates.length === 0) {
        return { isValid: false, error: 'No dates data' };
      }

      // Process close price data for line chart
      const closePriceData = [];
      for (let i = 0; i < Math.min(ohlc.length, dates.length); i++) {
        const item = ohlc[i];
        const date = dates[i];
        
        if (!item || !date || typeof item.close !== 'number' || isNaN(item.close)) {
          continue;
        }
        
        const timestamp = new Date(date).getTime();
        if (isNaN(timestamp)) continue;

        closePriceData.push({
          x: timestamp,
          y: Number(item.close.toFixed(2)),
        });
      }

      // Process prediction data
      let predictionData = [];
      if (last_week_data && Array.isArray(last_week_data) && last_week_data.length > 0) {
        // Use all available prediction data instead of limiting to 10
        for (const item of last_week_data) {
          if (!item || !item.date || typeof item.elu !== 'number' || isNaN(item.elu)) {
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
        isValid: closePriceData.length > 0,
        closePriceData,
        predictionData,
      };
    } catch (error) {
      console.error('Error processing line chart data:', error);
      return { isValid: false, error: 'Data processing error' };
    }
  }, [ohlc, dates, last_week_data]);

  // Early return if data is invalid
  if (!chartData.isValid) {
    return (
      <div className="flex justify-center items-center h-full">
        <p className="text-gray-500">Loading line chart data...</p>
      </div>
    );
  }

  const { closePriceData, predictionData } = chartData;

  // Line chart specific options
  const options: ApexOptions = {
    chart: {
      type: "line",
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
      text: "Close Price Line Chart",
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
      width: [3, 6], // Normal line, thicker prediction
      curve: 'smooth'
    },
    markers: {
      size: [10, 10], // Small markers for close price, large markers for prediction (matching candlestick chart)
      colors: ['#00E396', '#3B82F6'],
      strokeColors: ['#fff', '#fff'],
      strokeWidth: [2, 4],
      shape: 'circle',
      hover: {
        size: 20
      }
    },
    tooltip: {
      shared: true,
      intersect: false,
      y: {
        formatter: function(val) {
          return val ? `$${Number(val).toFixed(2)}` : '$0.00';
        }
      }
    },
    colors: ['#00E396', '#3B82F6'], // Green for close price, red for prediction
    grid: {
      borderColor: '#e7e7e7',
      strokeDashArray: 3
    }
  };

  // Build series for line chart
  const series = [];

  // Add close price line
  if (closePriceData && closePriceData.length > 0) {
    series.push({
      name: 'Close Price',
      data: closePriceData,
    });
  }

  // Add prediction line
  if (predictionData && predictionData.length > 0) {
    series.push({
      name: 'ELU Prediction',
      data: predictionData,
    });
  }

  return (
    <div className="w-full h-full" style={{ minHeight: `${height}px` }}>
      <ApexChart
        options={options}
        series={series}
        type="line"
        height="100%"
      />
    </div>
  );
};

export default React.memo(LineChart);
