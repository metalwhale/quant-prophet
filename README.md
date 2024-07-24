# quant-prophet
Quantitative strategy using magic based on faith and luck

## Local development
### Startup
1. Create a `.env` file by copying from [`./.env.local`](./.env.local):
    ```bash
    cp .env.local .env
    ```
    and then fill in the variables with appropriate values in the `.env` file.
2. Start the containers:
    ```bash
    docker compose up --build --remove-orphans -d
    ```

### Collect the data
#### US stock data
1. Get inside the container:
    ```bash
    docker compose exec collector bash
    ```
2. Run the command:
    ```bash
    python scripts/data_collector/yahoo/collector.py download_data --source_dir /usr/src/quant-prophet/data/stock/input/us --start 1900-01-01 --end 2024-12-31 --delay 1 --interval 1d --region US
    ```

Alternatively, you can download the US stock data from [this mirror URL](https://huggingface.co/datasets/metalwhale/stock_data/blob/e3c9912/us.zip) (updated as of 2024/07/24)

#### List of S&P 500 companies
1. Run the command (outside the container):
    ```bash
    curl https://raw.githubusercontent.com/datasets/s-and-p-500-companies/c7fb58f/data/constituents.csv -o ./data/stock/input/sp500.csv
    ```

### Train the agent
1. Get inside the container:
    ```bash
    docker compose exec trainer bash
    ```
2. Run the command:
    ```bash
    cd trainer/
    nohup python3 train.py &
    ```
