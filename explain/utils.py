import pandas as pd

def read_any(file, header):
    if file.endswith('.csv') or file.endswith('.tsv'):
        if header:
            df = pd.read_csv(file, header=0)
        else:
            df = pd.read_csv(file)
    elif file.endswith('.json'):
        df = pd.read_json(file)
    elif file.endswith('.xml'):
        df = pd.read_xml(file)
    elif file.endswith('.xls') or file.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.endswith('.hdf'):
        df = pd.read_hdf(file)
    elif file.endswith('.sql'):
        df = pd.read_sql(file)
    else:
        raise ValueError(f'Unsupported filetype: {file}')
    return df
