import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency


class HypothesisTester:
    def __init__(self, df):
        self.df = df.copy()
        self.results = {}

    def _calculate_claim_frequency(self, group):
        return (group['totalclaims'] > 0).mean()

    def _calculate_claim_severity(self, group):
        return group.loc[group['totalclaims'] > 0, 'totalclaims'].mean()

    def _calculate_margin(self, group):
        return (group['totalpremium'] - group['totalclaims']).mean()

    def test_province_risk_difference(self):
        groups = self.df.groupby("province")
        frequencies = [self._calculate_claim_frequency(g) for _, g in groups]

        if len(frequencies) < 2:
            self.results['province_risk'] = "Not enough unique provinces"
            return

        # Convert province column into contingency table
        contingency = pd.crosstab(self.df["province"], self.df["totalclaims"] > 0)
        chi2, p, _, _ = chi2_contingency(contingency)
        self.results["province_risk"] = {"p_value": p, "reject_H0": p < 0.05}

    def test_zipcode_risk_difference(self):
        grouped = self.df.groupby("postalcode")
        zip_codes = list(grouped.groups.keys())

        if len(zip_codes) < 2:
            self.results['zip_risk'] = "Not enough zip codes"
            return

        sample1 = self.df[self.df["postalcode"] == zip_codes[0]]['totalclaims']
        sample2 = self.df[self.df["postalcode"] == zip_codes[1]]['totalclaims']

        stat, p = ttest_ind(sample1, sample2, equal_var=False, nan_policy='omit')
        self.results["zip_risk"] = {"p_value": p, "reject_H0": p < 0.05}

    def test_zipcode_margin_difference(self):
        grouped = self.df.groupby("postalcode")
        margins = [self._calculate_margin(g) for _, g in grouped]
        if len(margins) < 2:
            self.results['zip_margin'] = "Not enough zip codes"
            return

        zip_codes = list(grouped.groups.keys())[:2]
        sample1 = self.df[self.df["postalcode"] == zip_codes[0]]
        sample2 = self.df[self.df["postalcode"] == zip_codes[1]]
        margin1 = sample1['totalpremium'] - sample1['totalclaims']
        margin2 = sample2['totalpremium'] - sample2['totalclaims']

        stat, p = ttest_ind(margin1, margin2, equal_var=False, nan_policy='omit')
        self.results["zip_margin"] = {"p_value": p, "reject_H0": p < 0.05}

    def test_gender_risk_difference(self):
        male = self.df[self.df["gender"].str.lower() == "male"]
        female = self.df[self.df["gender"].str.lower() == "female"]

        if len(male) == 0 or len(female) == 0:
            self.results['gender_risk'] = "Not enough male/female samples"
            return

        freq_m = self._calculate_claim_frequency(male)
        freq_f = self._calculate_claim_frequency(female)

        stat, p = ttest_ind(
            (male["totalclaims"] > 0).astype(int),
            (female["totalclaims"] > 0).astype(int),
            equal_var=False
        )
        self.results["gender_risk"] = {"p_value": p, "reject_H0": p < 0.05}

    def run_all_tests(self):
        self.test_province_risk_difference()
        self.test_zipcode_risk_difference()
        self.test_zipcode_margin_difference()
        self.test_gender_risk_difference()
        return self.results
