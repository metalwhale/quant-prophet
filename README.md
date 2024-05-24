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
1. Get inside the container:
    ```bash
    docker compose exec -it collector bash
    ```
2. Run the command:
    ```bash
    python scripts/data_collector/yahoo/collector.py download_data --source_dir /usr/src/quant-prophet/data/us_data --start 1900-01-01 --end 2024-12-31 --delay 1 --interval 1d --region US
    ```

### Train the agent
1. Get inside the container:
    ```bash
    docker compose exec -it trainer bash
    ```
