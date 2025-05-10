"use client";
import React, { useState } from "react";
import dynamic from "next/dynamic";
import { OHLCType, resultType } from "@/types";
import { ApexOptions } from "apexcharts";

// Dynamically import ApexCharts with SSR disabled for Next.js
const ApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

const PredictionChart = ({
  predictions,
  height,
  dates,
  lastWeekData,
}: {
  dates: string[];
  predictions: { tanh?: number; elu: number; actual?: number }[];
  height?: number;
  lastWeekData?: { date: string; elu: number; tanh: number; actual: number }[];
}) => {
  // Add state to track tanh visibility
  const [showTanh, setShowTanh] = useState(true);

  const options: ApexOptions = {
    xaxis: {
      type: "category",
    },
    yaxis: {
      tooltip: {
        enabled: true,
      },
    },
  };

  const tanh = predictions.filter((prediction) => prediction.tanh);
  const actual = predictions.filter(
    (prediction) => prediction.actual !== undefined
  );
  console.log(predictions, "predictions");
  const dateSeries = dates || [];
  console.log(actual.length);
  console.log(dates, "dates");
  const series = [
    {
      name: "Predicted L'Air Liquide Close Price",
      data: predictions.map((prediction, index) => ({
        x: dates[index].split("T")[0],
        y: prediction.elu.toFixed(2),
      })),
    },
  ];
  if (actual.length > 0) {
    console.log(predictions);
    series.push({
      name: "Actual Close Price",
      data: predictions.map((prediction, index) => ({
        x: dates[index].split("T")[0], // Remove time by splitting at "T" and taking the date part
        y: prediction.actual!.toFixed(2),
      })),
    });
  }
  console.log(dates);
  // Only add tanh series if toggle is on and tanh data exists
  if (showTanh && tanh.length > 0) {
    series.push({
      name: "TanH Close Price",
      data: predictions.map((prediction, index) => ({
        x: dates[index].split("T")[0],
        y: prediction.tanh!.toFixed(2),
      })),
    });
  }

  options.chart = {
    ...options.chart,
    type: "line",
    height: height || 350,

    toolbar: {
      show: true,
    },
  };

  options.markers = {
    size: 5,
    colors: ["#0A0F17"],
    strokeColors: "#fff",
    strokeWidth: 1,
    hover: {
      size: 6,
    },
  };

  options.grid = {
    show: true,
    borderColor: "#e7e7e7",
    strokeDashArray: 3,
    position: "back",
  };

  if (!height) {
    height = 700;
  }

  // Check if tanh data exists to show the toggle
  const hasTanhData = tanh.length > 0;

  return (
    <div className="flex flex-col justify-center items-center h-full">
      {/* Tanh Toggle Switch */}
      {hasTanhData && (
        <div className="flex items-center mb-4 self-end">
          <span className="mr-2 text-sm font-medium">Show TanH</span>
          <div 
            className={`relative w-11 h-6 cursor-pointer rounded-full transition-colors duration-200 ease-in-out ${showTanh ? 'bg-blue-600' : 'bg-gray-200'}`}
            onClick={() => setShowTanh(prev => !prev)}
          >
            <span 
              className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform duration-200 ease-in-out ${showTanh ? 'transform translate-x-5' : ''}`}
            ></span>
          </div>
        </div>
      )}

      <div className={`w-full h-full  min-h-[${height}px]`}>
        <ApexChart
          options={options}
          series={series}
          type="line"
          height="100%"
        />
      </div>
    </div>
  );
};

export default React.memo(PredictionChart);
