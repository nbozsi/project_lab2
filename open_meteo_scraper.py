import requests
import datetime
import polars as pl


def get_archive_forecast(lat=47.4979, lon=19.0402, variables=None):
    """
    Fetch tomorrow's weather forecast from Open-Meteo for given coordinates.
    Default is Budapest (47.4979 N, 19.0402 E).
    """
    if variables is None:
        # some default variables
        variables = [
            "temperature_2m",
            "precipitation",
            "weathercode",
            "windspeed_10m",
        ]

    # Calculate tomorrow's date in YYYY-MM-DD
    startdate = "2021-03-23"
    enddate = "2024-12-31"

    # Build API request
    url = "https://api.open-meteo.com/v1/forecast"
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(variables),
        # "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "timezone": "auto",
        "start_date": startdate,
        "end_date": enddate,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Print hourly data
    print("\nHourly forecast:")
    hours = data.get("hourly", {})
    return pl.DataFrame(hours)
    print(hours)
    if hours:
        for i, time in enumerate(hours["time"]):
            print(
                f"  {time}: {hours['temperature_2m'][i]} °C, "
                f"{hours['precipitation'][i]} mm, "
                f"wind {hours['windspeed_10m'][i]} km/h, "
                f"weathercode {hours['weathercode'][i]}"
            )


if __name__ == "__main__":
    df = get_archive_forecast()

    print(df)
    df.write_parquet("data/weather_data.parquet")
