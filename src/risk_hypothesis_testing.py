import pandas as pd
from scipy.stats import ttest_ind
import os

def load_data(path='../data/cleaned/machine_learning_rating_cleaned.csv'):
    return pd.read_csv(path)

def compute_claim_metrics(df):
    df['claim_occurred'] = df['totalclaims'] > 0
    df['margin'] = df['totalpremium'] - df['totalclaims']
    return df

def t_test(group1, group2, metric):
    stat, pval = ttest_ind(group1[metric].dropna(), group2[metric].dropna(), equal_var=False)
    return stat, pval

def test_gender_difference(df):
    print("🧪 Gender-Based Risk Testing:")
    male = df[df['gender'].str.lower() == 'male']
    female = df[df['gender'].str.lower() == 'female']
    for metric in ['claim_occurred', 'totalclaims', 'margin']:
        stat, p = t_test(male, female, metric)
        print(f" - {metric}: p = {p:.4f} | {'Reject H₀' if p < 0.05 else 'Fail to Reject H₀'}")

def test_province_difference(df):
    print("\n🧪 Province-Based Risk Testing:")
    top2 = df['province'].value_counts().index[:2]
    p1 = df[df['province'] == top2[0]]
    p2 = df[df['province'] == top2[1]]
    for metric in ['claim_occurred', 'totalclaims', 'margin']:
        stat, p = t_test(p1, p2, metric)
        print(f" - {top2[0]} vs {top2[1]} | {metric}: p = {p:.4f} | {'Reject H₀' if p < 0.05 else 'Fail to Reject H₀'}")

def test_zipcode_difference(df):
    print("\n🧪 ZipCode-Based Risk Testing:")
    top2 = df['postalcode'].value_counts().index[:2]
    z1 = df[df['postalcode'] == top2[0]]
    z2 = df[df['postalcode'] == top2[1]]
    for metric in ['claim_occurred', 'totalclaims', 'margin']:
        stat, p = t_test(z1, z2, metric)
        print(f" - {top2[0]} vs {top2[1]} | {metric}: p = {p:.4f} | {'Reject H₀' if p < 0.05 else 'Fail to Reject H₀'}")

def run_all_tests():
    df = load_data()
    df = compute_claim_metrics(df)
    test_gender_difference(df)
    test_province_difference(df)
    test_zipcode_difference(df)
