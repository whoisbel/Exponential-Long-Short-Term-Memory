"use client";
import React from "react";
import dynamic from "next/dynamic";
import { resultType } from "@/types";
import { ApexOptions } from "apexcharts";

// Dynamically import ApexCharts with SSR disabled for Next.js
const ApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

const LineChart = ({
  elu,
  tanh,
  height,
}: {
  elu: resultType;
  tanh: resultType;
  height?: number;
}) => {
  const maxDataPointsToShow = 50; // Limit the number of labels to avoid clutter
  const totalDataPoints = elu.predictions.length;
  const skipInterval = Math.ceil(totalDataPoints / maxDataPointsToShow);

  const xAxisLabels = elu.predictions.map((_, index) =>
    index % skipInterval === 0 ? index + 1 : ""
  );

  const options: ApexOptions = {
    chart: {
      type: "line",
      height: 400,
      toolbar: {
        show: true, // Enable zoom, pan, and download options
      },
    },
    stroke: {
      curve: "smooth",
      width: 2,
    },
    markers: {
      size: 3,
      hover: {
        size: 5,
      },
    },
    xaxis: {
      categories: xAxisLabels,
      title: {
        text: "Data Points",
        style: {
          fontSize: "12px",
          fontWeight: "bold",
        },
      },
      labels: {
        rotate: -45,
        style: {
          fontSize: "10px",
        },
      },
    },
    yaxis: {
      title: {
        text: "Values",
        style: {
          fontSize: "12px",
          fontWeight: "bold",
        },
      },
    },
    legend: {
      position: "top",
      horizontalAlign: "center",
      fontSize: "12px",
    },
    tooltip: {
      shared: true,
      intersect: false,
      y: {
        formatter: (value) => value.toFixed(2), // Format values for better readability
      },
    },
    grid: {
      borderColor: "#e7e7e7",
      strokeDashArray: 5,
    },
  };

  const series = [
    {
      name: "ELU Predictions",
      data: elu.predictions.length > 0 ? elu.predictions : [0],
    },
    {
      name: "Tanh Predictions",
      data: tanh.predictions.length > 0 ? tanh.predictions : [0],
    },
    {
      name: "Actual",
      data: elu.actuals.length > 0 ? elu.actuals : [0],
    },
  ];

  if (!height) {
    height = 700;
  }

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

export default React.memo(LineChart);
