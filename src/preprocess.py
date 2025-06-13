import pandas as pd
import os

class PreProcessData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def observe_first_rows(self, sample_size=2048):
        import csv
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                sample = f.read(sample_size)
                sniffer = csv.Sniffer()
                dialect = sniffer.sniff(sample)
                print(f"\U0001f9e0 Detected delimiter: '{dialect.delimiter}'")
                return dialect.delimiter
        except Exception as e:
            print(f" Failed to detect delimiter: {e}")
            return None

    def load_data(self, delimiter=None, chunksize=None):
        try:
            if delimiter is None:
                delimiter = self.detect_delimiter() or ','
            print(f" Using delimiter: '{delimiter}'")

            if chunksize:
                chunk_list = []
                for chunk in pd.read_csv(self.filepath, delimiter=delimiter, chunksize=chunksize, low_memory=False):
                    chunk_list.append(chunk)
                self.df = pd.concat(chunk_list, ignore_index=True)
            else:
                self.df = pd.read_csv(self.filepath, delimiter=delimiter, low_memory=False)

            print(f"✅ Data loaded successfully with shape: {self.df.shape}")
        except Exception as e:
            print(f" Error loading data: {e}")

    def observe_data(self):
        try:
            print(f"\n📊 Shape: {self.df.shape}\n")
            print("🔍 Data Types:")
            print(self.df.dtypes, "\n")
            print("🔍 Missing Values:")
            print(self.df.isnull().sum().sort_values(ascending=False), "\n")
            print("🔍 Descriptive Stats (Numerical):")
            print(self.df.describe(include='number'), "\n")
            print("🔍 Descriptive Stats (Categorical):")
            print(self.df.describe(include='object'), "\n")
            print("🔍 Unique Values:")
            for col in self.df.columns:
                print(f" - {col}: {self.df[col].nunique()} unique")
            print("\n🔍 Sample Rows:")
            print(self.df.sample(5))
        except Exception as e:
            print(f"Observation error: {e}")

    def clean_column_names(self):
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')

    def add_loss_ratio(self):
        try:
            self.df['loss_ratio'] = self.df['totalclaims'] / self.df['totalpremium']
        except Exception as e:
            print(f" Error calculating loss_ratio: {e}")

    def convert_transaction_month(self):
        if 'transactionmonth' in self.df.columns:
            self.df['transactionmonth'] = pd.to_datetime(self.df['transactionmonth'], errors='coerce')

    def clean(self):
        try:
            self.clean_column_names()
            self.df.drop(columns=[
                'numberofvehiclesinfleet', 'crossborder', 
                'rebuilt', 'converted', 'writtenoff'
            ], inplace=True, errors='ignore')
            self.df = self.df.copy()
            self.df['bank'] = self.df['bank'].fillna('Unknown')
            self.df['gender'] = self.df['gender'].fillna('Not specified')
            self.df['customvalueestimate'] = self.df['customvalueestimate'].fillna(0)
            self.df['covertype'] = self.df['covertype'].str.strip().str.lower()
            self.df['covercategory'] = self.df['covercategory'].str.strip().str.lower()
            self.add_loss_ratio()
            self.convert_transaction_month()
            print("✅ Cleaning complete")
        except Exception as e:
            print(f"Cleaning error: {e}")

    def save_cleaned(self, output_path):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.df.to_csv(output_path, index=False)
            print(f"✅ Cleaned data saved to: {output_path}")
        except Exception as e:
            print(f"Save error: {e}")
