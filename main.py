import pandas as pd

# params
DATA_PATH = 'data/sample_water_data.xlsx'
SHEET = 'Sheet2'


if __name__ == '__main__':
    data = pd.read_excel(DATA_PATH, sheet_name=SHEET)
    print(data)
