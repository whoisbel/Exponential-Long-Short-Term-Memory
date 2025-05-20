# ELU-LSTM

```bash
# Backend
cd backend
pip install pipenv
pipenv install
pipenv run python src/scripts/train_app.py --all-configs --dataset datasets/sample-data.csv

# Optional arguments:
# --all-configs: Train using all configurations from config.json
# --config CONFIG_NAME: Use a specific configuration from config.json
# --dataset DATASET_PATH: Path to the dataset CSV file (default: datasets/air_liquide.csv)

To activate this project's virtualenv, run pipenv shell.
Alternatively, run a command inside the virtualenv with pipenv run.
Installing dependencies from Pipfile.lock (2110f5)...

# Frontend
cd frontend
npm install
npm run dev
