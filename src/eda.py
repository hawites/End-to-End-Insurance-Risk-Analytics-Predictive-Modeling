import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class EDAVisualizer:
    def __init__(self, df):
        self.df = df

    def distribution_plots(self):
        numeric = ['totalpremium', 'totalclaims', 'loss_ratio']
        for col in numeric:
            if self.df[col].dropna().empty:
                print(f"⚠️ Skipping plot for {col}: no data.")
                continue
            plt.figure()
            sns.histplot(self.df[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.show()

    def categorical_distributions(self):
        for col in ['province', 'gender', 'vehicletype']:
            if self.df[col].dropna().nunique() == 0:
                print(f"⚠️ Skipping plot for {col}: no unique values.")
                continue
            plt.figure()
            sns.countplot(y=col, data=self.df, order=self.df[col].value_counts().index)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.show()

    def loss_ratio_grouped(self, by='province'):
        filtered = self.df[self.df['loss_ratio'] > 0]
        if filtered.empty:
            print(f"⚠️ Skipping loss ratio by {by}: no data after filtering.")
            return
        grouped = filtered.groupby(by)['loss_ratio'].mean().sort_values()
        plt.figure()
        grouped.plot(kind='barh', title=f"Average Loss Ratio by {by.title()}")
        plt.xlabel("Loss Ratio")
        plt.tight_layout()
        plt.show()

    def temporal_trends(self):
        if 'transactionmonth' not in self.df.columns or self.df['transactionmonth'].dropna().empty:
            print("⚠️ Skipping temporal trends: no valid transaction dates.")
            return
        trend = self.df.groupby('transactionmonth')[['totalpremium', 'totalclaims']].sum()
        if trend.empty:
            print("⚠️ Skipping temporal trends: no data grouped by month.")
            return
        trend.plot(title="Total Premium vs Total Claims Over Time", figsize=(10, 5))
        plt.ylabel("ZAR")
        plt.tight_layout()
        plt.show()

    def correlation_matrix(self):
        numeric = self.df.select_dtypes(include='number')
        if numeric.empty:
            print("⚠️ Skipping correlation matrix: no numeric data.")
            return
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric.corr(), cmap='coolwarm', annot=False)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    def outlier_boxplots(self):
        for col in ['totalclaims', 'customvalueestimate']:
            if self.df[col].dropna().empty:
                print(f"⚠️ Skipping boxplot for {col}: no data.")
                continue
            plt.figure()
            sns.boxplot(x=self.df[col])
            plt.title(f"Boxplot of {col}")
            plt.tight_layout()
            plt.show()

    def top_vehicles_by_claims(self, min_count=100):
        grouped = self.df.groupby('make').agg(
            avg_claims=('totalclaims', 'mean'),
            count=('make', 'count')
        ).query(f'count >= {min_count}')

        if grouped.empty:
            print(f"⚠️ No vehicle make found with at least {min_count} entries.")
            return

        top = grouped.sort_values('avg_claims', ascending=False).head(10)

        plt.figure()
        top['avg_claims'].plot(kind='bar')
        plt.title(f"Top 10 Vehicle Makes by Avg Total Claims (min {min_count} vehicles)")
        plt.ylabel("Avg Total Claims")
        plt.tight_layout()
        plt.show()
