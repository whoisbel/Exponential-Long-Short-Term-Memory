"use client";
import React from "react";
import dynamic from "next/dynamic";
import { OHLCType, resultType } from "@/types";
import { ApexOptions } from "apexcharts";

// Dynamically import ApexCharts with SSR disabled for Next.js
const ApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

const PredictionChart = ({
  predictions,
  height,
  dates,
}: {
  dates: string[];
  predictions: { tanh?: number; elu: number; actual?: number }[];
  height?: number;
}) => {
  const options: ApexOptions = {
    xaxis: {
      type: "datetime",
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
  const dateSeries = dates || [];
  console.log(actual.length);
  const series = [
    {
      name: "Predicted L'Air Liquide Close Price",
      data: predictions.map((prediction, index) => ({
        x: dates[index],
        y: prediction.elu.toFixed(2),
      })),
    },
  ];
  if (actual.length > 0) {
    console.log(predictions);
    series.push({
      name: "Actual Close Price",
      data: predictions.map((prediction, index) => ({
        x: dates[index],
        y: prediction.actual!.toFixed(2),
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
    size: 4,
    colors: ["#FFA41B"],
    strokeColors: "#fff",
    strokeWidth: 1,
    hover: {
      size: 6,
    },
  };

  options.grid = {
    show: true,
    borderColor: "#e7e7e7",
    strokeDashArray: 4,
    position: "back",
  };

  if (!height) {
    height = 700;
  }

  const trial = {
    series: [
      {
        data: [],
      },
    ],
    options: {
      chart: {
        type: "line",
        height: 350,
      },

      yaxis: {
        tooltip: {
          enabled: true,
        },
      },
    },
  };

  return (
    <div className="flex justify-center items-center  h-full">
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
