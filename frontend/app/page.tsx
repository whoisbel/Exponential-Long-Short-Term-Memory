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
  const [isDataset, setIsDataset] = useState(false);
  useEffect(() => {
    async function fetchPredictions() {
      setIsLoading(true);
      const res = await fetch(
        `http://localhost:8000/${
          isDataset ? "predict_with_dataset" : "predict-next-month"
        }`
      ); //ari ilisi
      const data = await res.json();
      console.log(data);
      setPredictions(data.predicted_values);
      if (data.base_data) {
        setBaseData([]);
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
      console.log(data.base_data);
      setIsLoading(false);
    }
    fetchPredictions();
  }, [isDataset]);

  function getDates() {
    const dates: string[] = [];
    if (baseData.length <= 0) {
      return dates;
    }
    const lastBaseDataDate = baseData[baseData.length - 1].date!;
    for (let i = 0; i < 10; i++) {
      const date = new Date(lastBaseDataDate);
      date.setDate(date.getDate() + i);
      dates.push(date.toISOString());
    }
    return dates;
  }
  const dates = getDates();
  return (
    <main className="flex flex-col md:flex-col w-full  h-full  ">
      <div className="grid grid-cols-6 gap-3 h-full p-4">
        <div className="flex flex-col col-span-4 h-full bg-white p-2">
          <div className="text-2xl p-2 font-bold">L'Air Liquide S.A</div>
          <LineChart
            dates={
              baseData.map((bData) => bData.date || Date.now()) as string[]
            }
            ohlc={baseData}
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
              Predicted Stock Price of Air Liquide for the next 10 days
            </h2>
            <div className="w-[200px] h-[30px] bg-slate-400 flex rounded relativ z-0">
              <button
                onClick={() => setIsDataset(!isDataset)}
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
              <tbody className="text-center">
                {predictions.map((prediction, index) => (
                  <tr key={index}>
                    <td>{dates[index].split("T")[0]}</td>

                    <td>{prediction.elu.toFixed(2)}</td>
                    <td>
                      {index > 0
                        ? (prediction.elu - predictions[index - 1].elu).toFixed(
                            2
                          )
                        : "0.00"}
                    </td>
                    <td>
                      {index > 0
                        ? (
                            ((prediction.elu - predictions[index - 1].elu) /
                              predictions[index - 1].elu) *
                            100
                          ).toFixed(2)
                        : "0.00"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </main>
  );
}
