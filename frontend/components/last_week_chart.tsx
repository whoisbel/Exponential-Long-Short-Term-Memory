'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { ApexOptions } from 'apexcharts';

// Dynamically import ApexCharts with SSR disabled for Next.js
const ApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

interface LastWeekProps {
  predictions: {
    date: string;
    elu: number;
    tanh: number;
    actual: number;
  }[];
  predictionPeriod: {
    start: string;
    end: string;
  };
}

export default function LastWeekChart({ predictions, predictionPeriod }: LastWeekProps) {
  // Extract dates for x-axis
  const labels = predictions.map((pred) => pred.date);

  const options: ApexOptions = {
    chart: {
      height: 350,
      type: 'line',
      toolbar: {
        show: true,
      },
    },
    colors: ['rgb(53, 162, 235)'],
    dataLabels: {
      enabled: false,
    },
    stroke: {
      width: [3],
      curve: 'straight',
    },
    title: {
      text: `Last Week ELU Prediction (${predictionPeriod.start} to ${predictionPeriod.end})`,
      align: 'left',
    },
    grid: {
      row: {
        colors: ['#f3f3f3', 'transparent'],
        opacity: 0.5,
      },
    },
    xaxis: {
      categories: labels,
    },
    legend: {
      position: 'top',
    },
    tooltip: {
      y: {
        formatter: function(val) {
          return val.toFixed(2);
        }
      }
    }
  };

  const series = [
    {
      name: 'ELU Prediction',
      data: predictions.map(pred => pred.elu),
    }
  ];

  return (
    <div className="w-full">
      <div className="w-full h-[400px]">
        <ApexChart
          options={options}
          series={series}
          type="line"
          height="100%"
        />
      </div>
      <div className="mt-6">
        <h3 className="text-lg font-semibold">ELU Prediction Values</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 mt-2">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ELU</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {predictions.map((pred, idx) => {
                return (
                  <tr key={idx}>
                    <td className="px-6 py-4 whitespace-nowrap">{pred.date}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{pred.elu.toFixed(2)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}