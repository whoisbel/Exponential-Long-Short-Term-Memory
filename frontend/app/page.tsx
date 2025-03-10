"use client";
import Navbar from "@/components/Navbar";
import { modelConfigType, resultType } from "@/types";
import dynamic from "next/dynamic";
import { useState } from "react";

// Dynamically import LineChart to avoid server-side rendering issues
const LineChart = dynamic(() => import("../components/linechart"), {
  ssr: false,
});

export default function Home() {
  const [elu, setElu] = useState<resultType>({
    predictions: [],
    actuals: [],
    metrics: {
      rmse: 0,
      mae: 0,
      mse: 0,
      r2: 0,
    },
  });

  const [tanh, setTanh] = useState<resultType>({
    predictions: [],
    actuals: [],
    metrics: {
      rmse: 0,
      mae: 0,
      mse: 0,
      r2: 0,
    },
  });

  const [modelConfigInput, setModelConfigInput] = useState<modelConfigType>({
    model_name: "",
    sequence_length: 0,
    epochs: 0,
    train_split: 0,
    batch_size: 0,
    hidden_size: 0,
    learning_rate: 0,
  });

  const [file, setFile] = useState<File | null>(null); // File input state
  const [isLoading, setIsLoading] = useState(false); // Track loading state

  const onSubmit = () => {
    if (!file) {
      alert("Please upload a dataset file.");
      return;
    }

    setIsLoading(true); // Start loading
    const formData = new FormData();
    formData.append("file", file);
    Object.keys(modelConfigInput).forEach((key) => {
      formData.append(key, modelConfigInput[key] as unknown as Blob);
    });

    fetch("http://127.0.0.1:8000/train", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        setIsLoading(false); // Stop loading
        if (data.error) {
          console.error("Error:", data.error);
        } else {
          setElu(data.results.ELU);
          setTanh(data.results.Tanh);
        }
      })
      .catch((error) => {
        setIsLoading(false); // Stop loading
        console.error("Error:", error);
      });
  };

  return (
    <main className="flex flex-col md:flex-col w-full  h-full  ">
      <div className="grid grid-cols-6 gap-3 h-full p-4">
        <div className="flex flex-col col-span-4 h-full bg-white p-2">
          <div className="text-2xl p-2 font-bold">L'Air Liquide S.A</div>
          <LineChart elu={elu} tanh={tanh} />
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
            <LineChart elu={elu} tanh={tanh} height={500} />
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
