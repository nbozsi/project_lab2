import requests
import datetime
import os


def download_mavir_xlsx(chart_number: int, year: int):
    base_url = f"https://www.mavir.hu/rtdwweb/webuser/chart/{chart_number}/export"
    export_type = "xls"
    period_type = "min"
    period = 15

    from_time = int(datetime.datetime(year, 1, 1).timestamp() * 1000)
    to_time = int(datetime.datetime(year + 1, 1, 1).timestamp() * 1000)

    params = {"exportType": export_type, "fromTime": from_time, "toTime": to_time, "periodType": period_type, "period": period}

    response = requests.get(base_url, params=params)

    if response.status_code == 200:
        return response.content
    else:
        return None


excel_folder = "excels"
years = range(2019, 2025)
chart_numbers = {
    "PV": 11838,
    "real_time_aggregated": 4401,
    "wind": 11840,
    "hatar_aramlas": 5229,
    "rendszerterheles": 7678,
}
done = 0
missed = 0
success = 0
for chart, code in chart_numbers.items():
    chart_folder = os.path.join(excel_folder, chart)
    if not os.path.exists(chart_folder):
        os.mkdir(chart_folder)
    for year in years:
        save_path = os.path.join(chart_folder, f"mavir_{year}_{chart}.xls")
        if os.path.exists(save_path):
            print(f"{chart:^20} | {year} | ✔")
            done += 1
            continue
        table = download_mavir_xlsx(code, year)
        if table:
            with open(save_path, "wb") as file:
                file.write(table)
            success += 1
            print(f"{chart:^20} | {year} | ✓")
        else:
            missed += 1
            print(f"{chart:^20} | {year} | 𐄂")
print()
print(f"{done} already downloaded, {success} newly added, {missed} missed out of {len(years)*len(chart_numbers)}")
