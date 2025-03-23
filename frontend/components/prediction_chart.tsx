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
}: {
  predictions: { tanh: number; elu: number }[];
  height?: number;
}) => {
  const options: ApexOptions = {
    chart: {
      type: "candlestick",
      height: 350,
    },
    title: {
      text: "Prediction Chart",
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
  };
  console.log(
    predictions.map((prediction, index) => ({
      x: new Date(new Date().setDate(new Date().getDate() + index)).getTime(), // Example timestamp
      y: prediction.elu,
    })),
    ``
  );
  const series = [
    {
      name: "TANH Prediction",
      data: predictions.map((prediction, index) => ({
        x: new Date(new Date().setDate(new Date().getDate() + index)).getTime(),
        y: prediction.tanh,
      })),
    },
    {
      name: "ELU Prediction",
      data: predictions.map((prediction, index) => ({
        x: new Date(new Date().setDate(new Date().getDate() + index)).getTime(), // Example timestamp
        y: prediction.elu,
      })),
    },
  ];

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
      title: {
        text: "Prediction Chart",
        align: "left",
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
