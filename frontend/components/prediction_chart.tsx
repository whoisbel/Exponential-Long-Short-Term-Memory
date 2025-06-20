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
  lastWeekData,
}: {
  dates: string[];
  predictions: { elu: number; actual?: number }[];
  height?: number;
  lastWeekData?: { date: string; elu: number; actual: number }[];
}) => {
  const options: ApexOptions = {
    xaxis: {
      type: "category",
    },
    yaxis: {
      tooltip: {
        enabled: true,
      },
      labels: {
        formatter: function (val) {
          return val.toFixed(2);
        },
      },
    },
    tooltip: {
      y: {
        formatter: function (val) {
          return val.toFixed(2);
        },
      },
    },
  };

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

  return (
    <div className="flex flex-col justify-center items-center h-full">
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
