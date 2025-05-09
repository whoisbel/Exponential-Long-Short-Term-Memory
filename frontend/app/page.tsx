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
const LastWeekChart = dynamic(
  () => import("../components/last_week_chart"),
  {
    ssr: false,
  }
);

export default function Home() {
  const [isLoading, setIsLoading] = useState(false); // Track loading state
  const [activeTab, setActiveTab] = useState<'predictions' | 'lastWeek'>('predictions');

  const [predictions, setPredictions] = useState<
    {
      elu: number;
      tanh?: number;
      actual?: number;
    }[]
  >([]);
  const [lastWeekData, setLastWeekData] = useState<{
    predicted_values: {
      date: string;
      elu: number;
      tanh: number;
      actual: number;
    }[];
    prediction_period: {
      start: string;
      end: string;
    };
  }>({
    predicted_values: [],
    prediction_period: { start: "", end: "" }
  });
  const [baseData, setBaseData] = useState<OHLCType[]>([]);
  const [isDataset, setIsDataset] = useState(false);
  const [dates, setDates] = useState<string[]>([]);
  
  useEffect(() => {
    setIsLoading(true);
    async function fetchPredictions() {
      const res = await fetch(
        `http://localhost:8000/${
          isDataset ? "predict_with_dataset" : "predict-next-month"
        }`
      );
      const data = await res.json();
      console.log(data);
      setPredictions(data.predicted_values);
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
    
    setDates(getDates());
    console.log(dates, "dates");
  }, [isDataset]);

  // Fetch last week's prediction data
  useEffect(() => {
    async function fetchLastWeekData() {
      setIsLoading(true);
      try {
        const res = await fetch('http://localhost:8000/api/predict-last-week');
        const data = await res.json();
        if (!data.error) {
          setLastWeekData(data);
        } else {
          console.error("Error fetching last week data:", data.error);
        }
      } catch (error) {
        console.error("Failed to fetch last week predictions:", error);
      }
      setIsLoading(false);
    }
    
    // Only fetch when the tab is active to save resources
    if (activeTab === 'lastWeek') {
      fetchLastWeekData();
    }
  }, [activeTab]);

  function getDates() {
    const dates: string[] = [];
    let startDate = new Date();

    if (!isDataset) {
      // If "Actual" is selected, start from tomorrow
      startDate.setDate(startDate.getDate() + 1);
    } else if (baseData.length > 0) {
      // Otherwise, use the last date from baseData
      startDate = new Date(baseData[baseData.length - 1].date!);
    }

    let i = 0;
    while (dates.length < (isDataset ? 60 : 3)) {
      const date = new Date(startDate);
      date.setDate(startDate.getDate() + i);
      const dayOfWeek = date.getDay();
      if (dayOfWeek !== 0 && dayOfWeek !== 6) {
        // Exclude Sundays (0) and Saturdays (6)
        dates.push(date.toISOString());
      }
      i++;
    }

    return dates.map((date) => date.split("T")[0]);
  }
  
  return (
    <main className="flex flex-col md:flex-col w-full h-full">
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
          />
          <div className="mt-auto hidden">
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
        <div className="flex flex-col w-full shadow-lg p-4 col-span-2 bg-white">
          <div className="flex flex-col">
            <h2 className="text-lg font-semibold mb-4">Air Liquide Predictions</h2>
            
            {/* Tab Navigation */}
            <div className="flex mb-4 border-b">
              <button 
                className={`py-2 px-4 font-medium ${activeTab === 'predictions' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600 hover:text-blue-500'}`}
                onClick={() => setActiveTab('predictions')}
              >
                Future Predictions
              </button>
              <button 
                className={`py-2 px-4 font-medium ${activeTab === 'lastWeek' ? 'border-b-2 border-blue-500 text-blue-600' : 'text-gray-600 hover:text-blue-500'}`}
                onClick={() => setActiveTab('lastWeek')}
              >
                Last Week Validation
              </button>
            </div>
            
            {/* Content based on active tab */}
            {activeTab === 'predictions' && (
              <>
                <div className="flex items-center mb-4">
                  <span className="mr-3">Mode:</span>
                  <div className="w-[200px] h-[30px] bg-slate-400 flex rounded relative z-0">
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
                
                <div className="h-[400px] bg-white shadow-lg">
                  <PredictionChart
                    predictions={predictions}
                    height={1000}
                    dates={getDates()}
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
                          predictions.map((prediction, index) => (
                            <tr key={index}>
                              <td>{dates && dates[index] ? dates[index].split("T")[0] : "N/A"}</td>
                              <td>
                                {index === 0 && baseData.length > 0
                                  ? baseData[baseData.length - 1].close.toFixed(2)
                                  : prediction.elu.toFixed(2)}
                              </td>
                              <td>
                                {index > 0
                                  ? (
                                      prediction.elu - predictions[index - 1].elu
                                    ).toFixed(2)
                                  : index === 0 && baseData.length > 0
                                  ? (
                                      prediction.elu - baseData[baseData.length - 1].close
                                    ).toFixed(2)
                                  : "0.00"}
                              </td>
                              <td>
                                {index > 0
                                  ? (
                                      ((prediction.elu - predictions[index - 1].elu) /
                                        predictions[index - 1].elu) *
                                      100
                                    ).toFixed(2)
                                  : index === 0 && baseData.length > 0
                                  ? (
                                      ((prediction.elu - baseData[baseData.length - 1].close) /
                                        baseData[baseData.length - 1].close) *
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
              </>
            )}
            
            {/* Last Week Tab Content */}
            {activeTab === 'lastWeek' && (
              <div className="h-full">
                {isLoading ? (
                  <div className="flex items-center justify-center h-64">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
                  </div>
                ) : lastWeekData.predicted_values.length > 0 ? (
                  <LastWeekChart 
                    predictions={lastWeekData.predicted_values} 
                    predictionPeriod={lastWeekData.prediction_period} 
                  />
                ) : (
                  <div className="flex items-center justify-center h-64">
                    <p>No last week data available.</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
