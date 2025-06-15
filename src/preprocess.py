import pandas as pd
import os

class PreProcessData:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def detect_delimiter(self, sample_size=2048):
        import csv
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                sample = f.read(sample_size)
                dialect = csv.Sniffer().sniff(sample)
                print(f" Detected delimiter: '{dialect.delimiter}'")
                return dialect.delimiter
        except Exception as e:
            print(f" Failed to detect delimiter: {e}")
            return ','

    def load_data(self, delimiter='|', chunksize=100000):
        try:
            chunks = pd.read_csv(self.filepath, delimiter=delimiter, chunksize=chunksize, low_memory=False)
            self.df = pd.concat(chunks, ignore_index=True)
            print(f"✅ Data loaded successfully with shape: {self.df.shape}")
        except Exception as e:
            print(f" Error loading data: {e}")

    def clean_column_names(self):
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')

    def drop_high_missing_rows(self, critical_columns, max_missing=2):
        # Drop rows where more than 2 of the critical columns are missing
        self.df['missing_count'] = self.df[critical_columns].isna().sum(axis=1)
        before = len(self.df)
        self.df = self.df[self.df['missing_count'] <= max_missing].drop(columns=['missing_count'])
        dropped = before - len(self.df)
        print(f" Dropped {dropped} rows with >= {max_missing+1} missing values from critical fields")

    def convert_transaction_month(self):
        if 'transactionmonth' in self.df.columns:
            self.df['transactionmonth'] = pd.to_datetime(self.df['transactionmonth'], errors='coerce')

    def add_loss_ratio(self):
        self.df['loss_ratio'] = pd.to_numeric(self.df['totalclaims'], errors='coerce') / \
                                pd.to_numeric(self.df['totalpremium'], errors='coerce')

    def basic_filtering(self):
        initial_shape = self.df.shape[0]

        # Filter 1: Remove rows with totalpremium <= 0
        cond1 = self.df['totalpremium'] > 0
        if cond1.sum() / initial_shape > 0.90:
            self.df = self.df[cond1]
            print(f"✅ Filtered rows with totalpremium > 0 → Remaining: {self.df.shape[0]}")
        
        # Filter 2: Remove rows with totalclaims < 0
        cond2 = self.df['totalclaims'] >= 0
        if cond2.sum() / initial_shape > 0.90:
            self.df = self.df[cond2]
            print(f"✅ Filtered rows with totalclaims >= 0 → Remaining: {self.df.shape[0]}")
       
        # Filter 3: Keep only rows with valid loss_ratio
        if 'loss_ratio' in self.df.columns:
            cond3 = self.df['loss_ratio'].notna()
            if cond3.sum() / initial_shape > 0.90:
                self.df = self.df[cond3]
                print(f"✅ Filtered rows with valid loss_ratio → Remaining: {self.df.shape[0]}")
         
            # Filter 4: Remove top 1% outliers in loss_ratio
            try:
                threshold = self.df['loss_ratio'].quantile(0.99)
                cond4 = self.df['loss_ratio'] < threshold
                if cond4.sum() / self.df.shape[0] > 0.90:
                    self.df = self.df[cond4]
                    print(f"✅ Removed top 1% outliers in loss_ratio → Remaining: {self.df.shape[0]}")
              
            except:
                print(" Could not compute loss_ratio quantile — skipping outlier filter.")
        else:
            print(" 'loss_ratio' column not found — skipping related filters.")


    def fill_defaults(self):
        self.df['bank'] = self.df['bank'].fillna('Unknown')
        self.df['gender'] = self.df['gender'].fillna('Not specified')
        self.df['customvalueestimate'] = self.df['customvalueestimate'].fillna(0)

    def normalize_strings(self):
        for col in ['covertype', 'covercategory', 'gender']:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().str.lower()

    def drop_useless_columns(self):
        cols_to_drop = [
            'numberofvehiclesinfleet', 'crossborder',
            'rebuilt', 'converted', 'writtenoff'
        ]
        self.df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    def clean(self):
        try:
            print(" Starting cleaning process...")
            self.clean_column_names()

            self.drop_high_missing_rows(
                critical_columns=['totalclaims', 'totalpremium', 'gender', 'transactionmonth'],
                max_missing=2
            )

            self.fill_defaults()
            self.normalize_strings()
            self.drop_useless_columns()
            self.convert_transaction_month()
            self.add_loss_ratio()
            self.basic_filtering()

            print("✅ Cleaning complete")
        except Exception as e:
            print(f" Cleaning error: {e}")

    def save_cleaned(self, output_path):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.df.to_csv(output_path, index=False)
            print(f"✅ Cleaned data saved to: {output_path}")
        except Exception as e:
            print(f" Save error: {e}")
