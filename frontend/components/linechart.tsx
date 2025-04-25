"use client";
import React from "react";
import dynamic from "next/dynamic";
import { OHLCType, resultType } from "@/types";
import { ApexOptions } from "apexcharts";

// Dynamically import ApexCharts with SSR disabled for Next.js
const ApexChart = dynamic(() => import("react-apexcharts"), { ssr: false });

const LineChart = ({
  dates,
  ohlc,
  height,
}: {
  dates: string[];
  ohlc: OHLCType[];
  height?: number;
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
  };

  const series = [
    {
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

  if (!height) {
    height = 700;
  }

  
  return (
    <div className="flex justify-center items-center  h-full">
      <div className={`w-full h-full  min-h-[${height}px]`}>
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
