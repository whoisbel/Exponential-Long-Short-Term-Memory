export type modelConfigType = {
  ticker: string;
  startDate: string;
  endDate: string;
  sequence_length: number;
  epochs: number;
  train_split: number;
  batch_size: number;
  hidden_size: number;
  learning_rate: number;
};

export type resultType = {
  predictions: number[];
  actuals: number[];
  metrics: {
    rmse: number;
    mae: number;
    msa: number;
    r2: number;
  };
};
