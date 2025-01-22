"use client";
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
    <main className="flex flex-col md:flex-row w-full h-full gap-6 px-6 py-4">
      {isLoading && (
        <div className="absolute inset-0 bg-black bg-opacity-50 z-50 flex flex-col items-center justify-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
          <p className="mt-4 text-white text-lg font-semibold">
            Training the model...
          </p>
        </div>
      )}
      <div className="flex flex-col gap-4 bg-white shadow-lg rounded p-4 w-full md:w-1/3">
        <h2 className="text-lg font-semibold">Model Configuration</h2>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => {
            if (e.target.files && e.target.files.length > 0) {
              setFile(e.target.files[0]);
            }
          }}
          className="border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {[
          { label: "Sequence Length", key: "sequence_length", type: "number" },
          { label: "Epochs", key: "epochs", type: "number" },
          { label: "Train Split (%)", key: "train_split", type: "number" },
          { label: "Batch Size", key: "batch_size", type: "number" },
          { label: "Hidden Size", key: "hidden_size", type: "number" },
          { label: "Learning Rate", key: "learning_rate", type: "number" },
        ].map(({ label, key, type }) => (
          <div key={key} className="flex flex-col gap-1">
            <label htmlFor={key} className="text-sm font-medium">
              {label}
            </label>
            <input
              id={key}
              type={type}
              value={modelConfigInput[key]}
              onChange={(e) =>
                setModelConfigInput({
                  ...modelConfigInput,
                  [key]: e.target.value,
                })
              }
              className="border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        ))}
        <button
          onClick={onSubmit}
          className="bg-blue-500 text-white rounded px-4 py-2 shadow hover:bg-blue-600 transition-colors"
        >
          Train Model
        </button>
      </div>

      <div className="flex flex-col w-full md:w-2/3 bg-white shadow-lg rounded p-4">
        <h2 className="text-lg font-semibold">Model Results</h2>
        <LineChart elu={elu} tanh={tanh} />
        <div className="mt-6">
          <h3 className="text-lg font-medium mb-2">Evaluation Metrics</h3>
          <table className="w-full border-collapse border border-gray-300">
            <thead>
              <tr>
                <th className="border border-gray-300 px-4 py-2">Metric</th>
                <th className="border border-gray-300 px-4 py-2">ELU</th>
                <th className="border border-gray-300 px-4 py-2">Tanh</th>
              </tr>
            </thead>
            <tbody>
              {["rmse", "mae", "mse", "r2"].map((metric) => (
                <tr key={metric}>
                  <td className="border border-gray-300 px-4 py-2 capitalize">
                    {metric.toUpperCase()}
                  </td>
                  <td className="border border-gray-300 px-4 py-2">
                    {elu.metrics[metric].toFixed(4)}
                  </td>
                  <td className="border border-gray-300 px-4 py-2">
                    {tanh.metrics[metric].toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}
