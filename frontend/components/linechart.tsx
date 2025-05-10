"use client";
import React from "react";
import dynamic from "next/dynamic";
import { OHLCType, resultType } from "@/types";
import { ApexOptions } from "apexcharts";

// Dynamically import ApexCharts with SSR disabled for Next.js
const ApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

interface LastWeekDataItem {
  date: string;
  elu: number;
  tanh?: number;
  actual?: number | null;
}

const LineChart = ({
  dates,
  ohlc,
  height,
  last_week_data,
}: {
  dates: string[];
  ohlc: OHLCType[];
  height?: number;
  last_week_data?: LastWeekDataItem[];
}) => {
  const options: ApexOptions = {
    chart: {
      type: "candlestick",
      height: 350,
    },
    title: {
      text: "CandleStick Chart",
      align: "left",
    },
    xaxis: {
      type: "datetime",
    },
    yaxis: {
      tooltip: {
        enabled: true,
      },
    },
    legend: {
      show: true,
    },
    stroke: {
      width: [1, 3, 3], // Added third element for tanh line
    },
    markers: {
      size: [0, 4, 4], // Added third element for tanh line
      shape: 'circle',
      hover: {
        size: 6
      }
    },
  };

  const candlestickSeries = [
    {
      name: 'Candlestick',
      type: 'candlestick',
      data: ohlc.map((data, index) => ({
        x: new Date(dates[index]).getTime(),
        y: [
          data.open.toFixed(2),
          data.high.toFixed(2),
          data.low.toFixed(2),
          data.close.toFixed(2),
        ],
      })),
    },
  ];

  const series = candlestickSeries;

  // Add prediction series if last_week_data is provided
  if (last_week_data && last_week_data.length > 0) {
    // Add ELU prediction series
    const eluSeries = {
      name: 'ELU Prediction',
      type: 'line',
      data: last_week_data.map(item => ({
        x: new Date(item.date).getTime(),
        y: Number(item.elu).toFixed(2),
      })),
      markers: {
        size: 5,
        colors: ['#FF4560'],
        strokeColors: '#fff',
        strokeWidth: 2,
      },
      tooltip: {
        shared: true,
        intersect: false,
      },
      stroke: {
        width: 2,
        curve: 'smooth',
      },
      colors: ['#FF4560'],
      dataLabels: {
        enabled: false,
      },
    };
    
    series.push(eluSeries);

    // Add tanh prediction series if available
    if (last_week_data[0].tanh !== undefined) {
      const tanhSeries = {
        name: 'Tanh Prediction',
        type: 'line',
        data: last_week_data.map(item => ({
          x: new Date(item.date).getTime(),
          y: item.tanh ? Number(item.tanh).toFixed(2) : null,
        })),
        markers: {
          size: 5,
          colors: ['#00B746'],
          strokeColors: '#fff',
          strokeWidth: 2,
        },
        tooltip: {
          shared: true,
          intersect: false,
        },
        stroke: {
          width: 2,
          curve: 'smooth',
        },
        colors: ['#00B746'],
        dataLabels: {
          enabled: false,
        },
      };
      
      series.push(tanhSeries);
    }
  }

  if (!height) {
    height = 700;
  }

  
  return (
    <div className="flex justify-center items-center  h-full">
      <div className="w-full h-full" style={{ minHeight: `${height}px` }}>
        <ApexChart
          options={options}
          series={series}
          type="candlestick"
          height="100%"
        />
      </div>
    </div>
  );
};

export default React.memo(LineChart);
