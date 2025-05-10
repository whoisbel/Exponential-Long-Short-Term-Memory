"use client";
import Navbar from "@/components/Navbar";
import { modelConfigType, OHLCType, resultType } from "@/types";
import dynamic from "next/dynamic";
import { useEffect, useState } from "react";

// Dynamically import LineChart to avoid server-side rendering issues
const LineChart = dynamic(() => import("../components/linechart"), {
  ssr: false,
});
const PredictionChart = dynamic(
  () => import("../components/prediction_chart"),
  {
    ssr: false,
  }
);

export default function Home() {
  const [isLoading, setIsLoading] = useState(false); // Track loading state

  const [predictions, setPredictions] = useState<
    {
      elu: number;
      tanh?: number;
      actual?: number;
    }[]
  >([]);
  const [baseData, setBaseData] = useState<OHLCType[]>([]);
  const [lastWeekData, setLastWeekData] = useState<
    {
      date: string; 
      elu: number;
      tanh: number;
      actual: number;
    }[]
  >([]);
  const [isDataset, setIsDataset] = useState(false);
  useEffect(() => {
    setIsLoading(true);
    async function fetchPredictions() {
      const res = await fetch(
        `http://localhost:8000/${
          isDataset ? "predict_with_dataset" : "predict-next-month"
        }`
      ); //ari ilisi
      const data = await res.json();
      console.log(data);
      setPredictions(data.predicted_values ? data.predicted_values : []);
      
      // Handle last week data if available
      if (!isDataset && data.last_week_data) {
        setLastWeekData(data.last_week_data);
      } else {
        setLastWeekData([]);
      }
      
      if (data.base_data) {
        setBaseData((prev) => prev.slice(0, 0)); // Clear previous data
        data.base_data.map((bData: any) => {
          /*
          Date,Close,High,Low,Open,Volume
          */
          setBaseData((prevData) => [
            ...prevData,
            {
              date: bData[0],
              close: bData[1],
              high: bData[2],
              low: bData[3],
              open: bData[4],
              volume: bData[5],
            },
          ]);
        });
      }
      console.log(data.base_data, "hellooooo");
      setIsLoading(false);
    }
    fetchPredictions();
    console.log(dates, "dates");
  }, [isDataset]);

  function getDates() {
    const dates: string[] = [];
    if (baseData.length <= 0) {
      return dates;
    }
    const lastBaseDataDate = baseData[baseData.length - 1].date!;
    let i = 0;
    while (dates.length < 60) {
      const date = new Date(lastBaseDataDate);
      date.setDate(date.getDate() + i + 1);
      const dayOfWeek = date.getDay();
      if (dayOfWeek !== 0 && dayOfWeek !== 6) {
        // Exclude Sundays (0) and Saturdays (6)
        dates.push(date.toISOString());
      }
      i++;
    }

    return dates.map((date) => date.split("T")[0]);
  }
  const dates = getDates();
  return (
    <main className="flex flex-col md:flex-col w-full  h-full  ">
      <div className="grid grid-cols-6 gap-3 h-full p-4">
        <div className="flex flex-col col-span-4 h-full bg-white p-2">
          <div className="text-2xl p-2 font-bold">
            L'Air Liquide S.A{" "}
            {`$ ${
              baseData.length > 1 &&
              baseData[baseData.length - 1].close.toFixed(2)
            }`}{" "}
          </div>
          <LineChart
            dates={
              baseData.map((bData) => bData.date || Date.now()) as string[]
            }
            ohlc={baseData}
            last_week_data={lastWeekData}
          />
          <div className=" mt-auto hidden">
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
          <div className="flex flex-col">
            <h2 className="text-lg font-semibold">
              Predicted Stock Price of Air Liquide for the next{" "}
              {predictions.length} days
            </h2>
            <div className="w-[200px] h-[30px] bg-slate-400 flex rounded relativ z-0">
              <button
                onClick={() => setIsDataset((prev) => !prev)}
                className={`bg-blue-500 text-white px-4 rounded absolute w-[100px] h-[30px] transition-transform ease-in-out duration-700 ${
                  isDataset ? "translate-x-full" : "translate-x-0"
                }`}
              >
                {`${isDataset ? "Dataset" : "Actual"}`}
              </button>
            </div>
          </div>
          <div className="h-[400px] bg-white shadow-lg ">
            <PredictionChart
              predictions={predictions}
              height={1000}
              dates={getDates()}
              lastWeekData={lastWeekData}
            />
          </div>
          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Prediction Table</h3>
            <div className="max-h-[250px] overflow-y-auto">
              <table className="w-full border-collapse border border-gray-300">
                <thead className="sticky top-0 bg-white shadow">
                  <tr>
                    <th className="border border-gray-300 px-4 py-2">Date</th>
                    <th className="border border-gray-300 px-4 py-2">
                      LastPrice
                    </th>
                    <th className="border border-gray-300 px-4 py-2">Change</th>
                    <th className="border border-gray-300 px-4 py-2">
                      % Change
                    </th>
                  </tr>
                </thead>
                <tbody className="text-center">
                  {predictions &&
                    predictions.map((prediction, index) => {
                      // For first day prediction, get the last actual price from baseData
                      const lastActualPrice = index === 0 && baseData.length > 0 
                        ? baseData[baseData.length - 1].close 
                        : index > 0 ? predictions[index - 1].elu : null;
                      
                      // Calculate change and percent change
                      const change = lastActualPrice !== null
                        ? (prediction.elu - lastActualPrice).toFixed(2)
                        : "0.00";
                      
                      const percentChange = lastActualPrice !== null && lastActualPrice !== 0
                        ? ((prediction.elu - lastActualPrice) / lastActualPrice * 100).toFixed(2)
                        : "0.00";
                      
                      return (
                        <tr key={index}>
                          <td>{dates[index].split("T")[0]}</td>
                          <td>{prediction.elu.toFixed(2)}</td>
                          <td>{change}</td>
                          <td>{percentChange}%</td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
