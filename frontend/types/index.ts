export type modelConfigType = {
  model_name: string;
  ticker?: string;
  startDate?: string;
  endDate?: string;
  sequence_length: number;
  epochs: number;
  train_split: number;
  batch_size: number;
  hidden_size: number;
  learning_rate: number;
  [key: string]: string | number | undefined;
};

type metricType = {
  rmse: number;
  mae: number;

  r2: number;
  mse: number;
  [keyof: string]: number;
};

export type resultType = {
  predictions: number[];
  actuals: number[];
  metrics: metricType;
  [key: string]:
    | number[]
    | { rmse: number; mae: number; r2: number; mse: number };
};
export type OHLCType = {
  open: number;
  high: number;
  low: number;
  close: number;
};
