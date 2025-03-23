"use client";
import Navbar from "@/components/Navbar";
import { modelConfigType, OHLCType, resultType } from "@/types";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

// Dynamically import LineChart to avoid server-side rendering issues
const LineChart = dynamic(() => import("../components/linechart"), {
  ssr: false,
});

export default function Home() {
  const [isLoading, setIsLoading] = useState(false); // Track loading state

  const [predictions, setPredictions] = useState<{
    dates: string[];
    ohlc: OHLCType[];
  }>({
    dates: [],
    ohlc: [],
  });
  useEffect(() => {
    async function fetchPredictions() {
      setIsLoading(true);
      const res = await fetch("/api/predict"); //ari ilisi
      const data = await res.json();
      setPredictions(data);
      setIsLoading(false);
    }
    fetchPredictions();
  });

  return (
    <main className="flex flex-col md:flex-col w-full  h-full  ">
      <div className="grid grid-cols-6 gap-3 h-full p-4">
        <div className="flex flex-col col-span-4 h-full bg-white p-2">
          <div className="text-2xl p-2 font-bold">L'Air Liquide S.A</div>
          <LineChart dates={predictions.dates} ohlc={predictions.ohlc} />
          <div className=" mt-auto ">
            <ul className="flex mt-auto">
              <li className="p-2 hover:scale-105 cursor-pointer">1D</li>
              <li className="p-2 hover:scale-105 cursor-pointer">5D</li>
              <li className="p-2 hover:scale-105 cursor-pointer">1M</li>
              <li className="p-2 hover:scale-105 cursor-pointer">3M</li>
              <li className="p-2 hover:scale-105 cursor-pointer">6M</li>
              <li className="p-2 hover:scale-105 cursor-pointer">YTD</li>
              <li className="p-2 hover:scale-105 cursor-pointer">1Y</li>
              <li className="p-2 hover:scale-105 cursor-pointer">5Y</li>
            </ul>
          </div>
        </div>
        <div className="flex flex-col w-full   shadow-lg p-4 col-span-2 bg-white ">
          <h2 className="text-lg font-semibold">10 days Prediction</h2>
          <div className="h-full bg-white shadow-lg ">
            <LineChart
              dates={predictions.dates}
              ohlc={predictions.ohlc}
              height={500}
            />
          </div>
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Prediction Table</h3>
            <table className="w-full border-collapse border border-gray-300">
              <thead>
                <tr>
                  <th className="border border-gray-300 px-4 py-2">Date</th>
                  <th className="border border-gray-300 px-4 py-2">
                    LastPrice
                  </th>
                  <th className="border border-gray-300 px-4 py-2">Change</th>
                  <th className="border border-gray-300 px-4 py-2">% Change</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>No Data</td>
                </tr>
                <tr>
                  <td>No Data</td>
                </tr>
                <tr>
                  <td>No Data</td>
                </tr>
                <tr>
                  <td>No Data</td>
                </tr>
                <tr>
                  <td>No Data</td>
                </tr>
                <tr>
                  <td>No Data</td>
                </tr>
                <tr>
                  <td>No Data</td>
                </tr>
                <tr>
                  <td>No Data</td>
                </tr>
                <tr>
                  <td>No Data</td>
                </tr>
                <tr>
                  <td>No Data</td>
                </tr>
                <tr>
                  <td>No Data</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </main>
  );
}
