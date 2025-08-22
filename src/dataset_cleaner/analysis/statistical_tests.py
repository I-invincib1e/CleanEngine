#!/usr/bin/env python3
"""
Advanced Statistical Tests Module
Comprehensive statistical testing suite for data analysis.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact

warnings.filterwarnings("ignore")


class StatisticalTester:
    """Advanced statistical testing capabilities"""

    def __init__(self, df):
        self.df = df.copy()
        self.test_results = {}
        self.insights = []

    def test_normality(self, columns=None, alpha=0.05):
        """Test normality of numeric columns using multiple methods"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns

        normality_results = {}

        for col in columns:
            if col not in self.df.columns:
                continue

            data = self.df[col].dropna()
            if len(data) < 3:
                continue

            results = {}

            # Shapiro-Wilk test (good for small samples)
            if len(data) <= 5000:
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(data)
                    results["shapiro_wilk"] = {
                        "statistic": shapiro_stat,
                        "p_value": shapiro_p,
                        "is_normal": shapiro_p > alpha,
                    }
                except (ValueError, RuntimeWarning) as e:
                    # Data may be too small or have issues
                    pass

            # D'Agostino's normality test (good for larger samples)
            if len(data) >= 20:
                try:
                    dagostino_stat, dagostino_p = stats.normaltest(data)
                    results["dagostino"] = {
                        "statistic": dagostino_stat,
                        "p_value": dagostino_p,
                        "is_normal": dagostino_p > alpha,
                    }
                except (ValueError, RuntimeWarning) as e:
                    # Data may have issues
                    pass

            # Kolmogorov-Smirnov test
            try:
                # Compare with normal distribution with same mean and std
                ks_stat, ks_p = stats.kstest(
                    data, lambda x: stats.norm.cdf(x, data.mean(), data.std())
                )
                results["kolmogorov_smirnov"] = {
                    "statistic": ks_stat,
                    "p_value": ks_p,
                    "is_normal": ks_p > alpha,
                }
            except (ValueError, RuntimeWarning) as e:
                # Data may have issues
                pass

            # Anderson-Darling test
            try:
                ad_result = stats.anderson(data, dist="norm")
                # Use 5% significance level (index 2)
                critical_value = ad_result.critical_values[2]
                results["anderson_darling"] = {
                    "statistic": ad_result.statistic,
                    "critical_value": critical_value,
                    "is_normal": ad_result.statistic < critical_value,
                }
            except (ValueError, RuntimeWarning) as e:
                # Data may have issues
                pass

            # Consensus decision
            normal_votes = sum(
                1 for test in results.values() if test.get("is_normal", False)
            )
            total_tests = len(results)
            consensus_normal = (
                normal_votes > total_tests / 2 if total_tests > 0 else False
            )

            results["consensus"] = {
                "is_normal": consensus_normal,
                "confidence": normal_votes / total_tests if total_tests > 0 else 0,
            }

            normality_results[col] = results

            # Generate insights
            if not consensus_normal:
                self.insights.append(
                    f"{col} does not follow normal distribution (consider transformation)"
                )
            elif results["consensus"]["confidence"] < 0.7:
                self.insights.append(
                    f"{col} normality is uncertain - use non-parametric tests"
                )

        self.test_results["normality_tests"] = normality_results
        return normality_results

    def test_equal_variances(self, groups, alpha=0.05):
        """Test equality of variances between groups"""
        if len(groups) < 2:
            return None

        variance_results = {}

        # Levene's test (robust to non-normality)
        try:
            levene_stat, levene_p = stats.levene(*groups)
            variance_results["levene"] = {
                "statistic": levene_stat,
                "p_value": levene_p,
                "equal_variances": levene_p > alpha,
            }
        except (ValueError, RuntimeWarning) as e:
            # Groups may have issues
            pass

        # Bartlett's test (assumes normality)
        try:
            bartlett_stat, bartlett_p = stats.bartlett(*groups)
            variance_results["bartlett"] = {
                "statistic": bartlett_stat,
                "p_value": bartlett_p,
                "equal_variances": bartlett_p > alpha,
            }
        except (ValueError, RuntimeWarning) as e:
            # Groups may have issues
            pass

        # Fligner-Killeen test (non-parametric)
        try:
            fligner_stat, fligner_p = stats.fligner(*groups)
            variance_results["fligner_killeen"] = {
                "statistic": fligner_stat,
                "p_value": fligner_p,
                "equal_variances": fligner_p > alpha,
            }
        except (ValueError, RuntimeWarning) as e:
            # Groups may have issues
            pass

        return variance_results

    def anova_test(self, dependent_var, independent_var, alpha=0.05):
        """Perform one-way ANOVA test"""
        if (
            dependent_var not in self.df.columns
            or independent_var not in self.df.columns
        ):
            return None

        # Clean data
        clean_data = self.df[[dependent_var, independent_var]].dropna()
        if len(clean_data) < 3:
            return None

        # Group data
        groups = [
            group[dependent_var].values
            for name, group in clean_data.groupby(independent_var)
        ]
        groups = [g for g in groups if len(g) > 0]  # Remove empty groups

        if len(groups) < 2:
            return None

        anova_results = {}

        # One-way ANOVA
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            anova_results["one_way_anova"] = {
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant_difference": p_value < alpha,
            }

            # Effect size (eta-squared)
            ss_between = sum(
                len(g) * (np.mean(g) - np.mean(np.concatenate(groups))) ** 2
                for g in groups
            )
            ss_total = sum(
                (x - np.mean(np.concatenate(groups))) ** 2 for g in groups for x in g
            )
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            anova_results["one_way_anova"]["eta_squared"] = eta_squared

            # Generate insights
            if p_value < alpha:
                effect_size = (
                    "large"
                    if eta_squared > 0.14
                    else "medium" if eta_squared > 0.06 else "small"
                )
                self.insights.append(
                    f"Significant difference in {dependent_var} across {independent_var} groups (η² = {eta_squared:.3f}, {effect_size} effect)"
                )

        except (ValueError, RuntimeWarning) as e:
            # Groups may have issues
            pass

        # Test assumptions
        variance_test = self.test_equal_variances(groups, alpha)
        if variance_test:
            anova_results["variance_test"] = variance_test

        self.test_results[f"anova_{dependent_var}_by_{independent_var}"] = anova_results
        return anova_results

    def chi_square_test(self, var1, var2, alpha=0.05):
        """Perform chi-square test of independence"""
        if var1 not in self.df.columns or var2 not in self.df.columns:
            return None

        # Create contingency table
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])

        if contingency_table.size == 0:
            return None

        chi2_results = {}

        try:
            # Chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

            # Check minimum expected frequency assumption
            min_expected = expected.min()
            cells_below_5 = (expected < 5).sum()
            total_cells = expected.size

            chi2_results["chi_square"] = {
                "chi2_statistic": chi2_stat,
                "p_value": p_value,
                "degrees_of_freedom": dof,
                "significant_association": p_value < alpha,
                "min_expected_frequency": min_expected,
                "cells_below_5": cells_below_5,
                "percent_cells_below_5": (cells_below_5 / total_cells) * 100,
                "assumption_met": cells_below_5 / total_cells < 0.2
                and min_expected >= 1,
            }

            # Cramér's V (effect size)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            chi2_results["chi_square"]["cramers_v"] = cramers_v

            # Fisher's exact test for 2x2 tables
            if contingency_table.shape == (2, 2):
                try:
                    odds_ratio, fisher_p = fisher_exact(contingency_table)
                    chi2_results["fisher_exact"] = {
                        "odds_ratio": odds_ratio,
                        "p_value": fisher_p,
                        "significant": fisher_p < alpha,
                    }
                except:
                    pass

            # Generate insights
            if p_value < alpha:
                effect_size = (
                    "large"
                    if cramers_v > 0.5
                    else "medium" if cramers_v > 0.3 else "small"
                )
                self.insights.append(
                    f"Significant association between {var1} and {var2} (Cramér's V = {cramers_v:.3f}, {effect_size} effect)"
                )

            if not chi2_results["chi_square"]["assumption_met"]:
                self.insights.append(
                    f"Chi-square assumptions violated for {var1} vs {var2} - consider Fisher's exact test"
                )

        except (ValueError, RuntimeWarning) as e:
            # Contingency table may have issues
            pass

        self.test_results[f"chi_square_{var1}_vs_{var2}"] = chi2_results
        return chi2_results

    def correlation_tests(self, var1, var2, alpha=0.05):
        """Perform multiple correlation tests"""
        if var1 not in self.df.columns or var2 not in self.df.columns:
            return None

        # Clean data
        clean_data = self.df[[var1, var2]].dropna()
        if len(clean_data) < 3:
            return None

        correlation_results = {}

        # Pearson correlation (parametric)
        try:
            pearson_r, pearson_p = stats.pearsonr(clean_data[var1], clean_data[var2])
            correlation_results["pearson"] = {
                "correlation": pearson_r,
                "p_value": pearson_p,
                "significant": pearson_p < alpha,
            }
        except (ValueError, RuntimeWarning) as e:
            # Data may have issues
            pass

        # Spearman correlation (non-parametric)
        try:
            spearman_r, spearman_p = stats.spearmanr(clean_data[var1], clean_data[var2])
            correlation_results["spearman"] = {
                "correlation": spearman_r,
                "p_value": spearman_p,
                "significant": spearman_p < alpha,
            }
        except (ValueError, RuntimeWarning) as e:
            # Data may have issues
            pass

        # Kendall's tau (non-parametric, robust)
        try:
            kendall_tau, kendall_p = stats.kendalltau(
                clean_data[var1], clean_data[var2]
            )
            correlation_results["kendall"] = {
                "correlation": kendall_tau,
                "p_value": kendall_p,
                "significant": kendall_p < alpha,
            }
        except (ValueError, RuntimeWarning) as e:
            # Data may have issues
            pass

        # Generate insights
        for method, result in correlation_results.items():
            if result.get("significant", False):
                corr_strength = abs(result["correlation"])
                strength = (
                    "strong"
                    if corr_strength > 0.7
                    else "moderate" if corr_strength > 0.3 else "weak"
                )
                direction = "positive" if result["correlation"] > 0 else "negative"
                self.insights.append(
                    f"{strength.title()} {direction} {method} correlation between {var1} and {var2} (r = {result['correlation']:.3f})"
                )

        self.test_results[f"correlation_{var1}_vs_{var2}"] = correlation_results
        return correlation_results

    def t_tests(self, var1, var2=None, group_var=None, alpha=0.05):
        """Perform various t-tests"""
        t_test_results = {}

        if var2 is not None:
            # Paired t-test
            clean_data = self.df[[var1, var2]].dropna()
            if len(clean_data) >= 3:
                try:
                    t_stat, p_value = stats.ttest_rel(
                        clean_data[var1], clean_data[var2]
                    )

                    # Effect size (Cohen's d for paired samples)
                    diff = clean_data[var1] - clean_data[var2]
                    cohens_d = diff.mean() / diff.std()

                    t_test_results["paired_t_test"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < alpha,
                        "cohens_d": cohens_d,
                        "mean_difference": diff.mean(),
                    }

                    # Generate insights
                    if p_value < alpha:
                        effect_size = (
                            "large"
                            if abs(cohens_d) > 0.8
                            else "medium" if abs(cohens_d) > 0.5 else "small"
                        )
                        direction = "increase" if diff.mean() > 0 else "decrease"
                        self.insights.append(
                            f"Significant {direction} from {var2} to {var1} (d = {cohens_d:.3f}, {effect_size} effect)"
                        )

                except (ValueError, RuntimeWarning) as e:
                    t_test_results["paired_t_test_error"] = str(e)

        elif group_var is not None:
            # Independent samples t-test
            clean_data = self.df[[var1, group_var]].dropna()
            groups = clean_data.groupby(group_var)[var1].apply(list)

            if len(groups) == 2:
                group1, group2 = groups.iloc[0], groups.iloc[1]

                if len(group1) >= 2 and len(group2) >= 2:
                    try:
                        # Equal variances t-test
                        t_stat_equal, p_value_equal = stats.ttest_ind(
                            group1, group2, equal_var=True
                        )

                        # Unequal variances t-test (Welch's)
                        t_stat_unequal, p_value_unequal = stats.ttest_ind(
                            group1, group2, equal_var=False
                        )

                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(
                            (
                                (len(group1) - 1) * np.var(group1, ddof=1)
                                + (len(group2) - 1) * np.var(group2, ddof=1)
                            )
                            / (len(group1) + len(group2) - 2)
                        )
                        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std

                        t_test_results["independent_t_test"] = {
                            "equal_var": {
                                "t_statistic": t_stat_equal,
                                "p_value": p_value_equal,
                                "significant": p_value_equal < alpha,
                            },
                            "unequal_var": {
                                "t_statistic": t_stat_unequal,
                                "p_value": p_value_unequal,
                                "significant": p_value_unequal < alpha,
                            },
                            "cohens_d": cohens_d,
                            "group_means": {
                                groups.index[0]: np.mean(group1),
                                groups.index[1]: np.mean(group2),
                            },
                        }

                        # Test equal variances assumption
                        variance_test = self.test_equal_variances(
                            [group1, group2], alpha
                        )
                        if variance_test:
                            t_test_results["variance_test"] = variance_test

                        # Generate insights
                        p_val = (
                            p_value_equal
                            if variance_test
                            and variance_test.get("levene", {}).get(
                                "equal_variances", True
                            )
                            else p_value_unequal
                        )
                        if p_val < alpha:
                            effect_size = (
                                "large"
                                if abs(cohens_d) > 0.8
                                else "medium" if abs(cohens_d) > 0.5 else "small"
                            )
                            higher_group = (
                                groups.index[0]
                                if np.mean(group1) > np.mean(group2)
                                else groups.index[1]
                            )
                            self.insights.append(
                                f"Significant difference in {var1} between {group_var} groups (d = {cohens_d:.3f}, {effect_size} effect, {higher_group} higher)"
                            )

                    except (ValueError, RuntimeWarning) as e:
                        t_test_results["independent_t_test_error"] = str(e)

        if t_test_results:
            test_key = f"t_test_{var1}"
            if var2:
                test_key += f"_vs_{var2}"
            elif group_var:
                test_key += f"_by_{group_var}"

            self.test_results[test_key] = t_test_results

        return t_test_results

    def non_parametric_tests(self, var1, var2=None, group_var=None, alpha=0.05):
        """Perform non-parametric tests"""
        nonparam_results = {}

        if var2 is not None:
            # Wilcoxon signed-rank test (paired)
            clean_data = self.df[[var1, var2]].dropna()
            if len(clean_data) >= 3:
                try:
                    w_stat, p_value = stats.wilcoxon(clean_data[var1], clean_data[var2])

                    nonparam_results["wilcoxon_signed_rank"] = {
                        "statistic": w_stat,
                        "p_value": p_value,
                        "significant": p_value < alpha,
                    }

                    # Generate insights
                    if p_value < alpha:
                        median_diff = (
                            clean_data[var1].median() - clean_data[var2].median()
                        )
                        direction = "increase" if median_diff > 0 else "decrease"
                        self.insights.append(
                            f"Significant median {direction} from {var2} to {var1} (Wilcoxon test)"
                        )

                except (ValueError, RuntimeWarning) as e:
                    nonparam_results["wilcoxon_error"] = str(e)

        elif group_var is not None:
            # Mann-Whitney U test or Kruskal-Wallis test
            clean_data = self.df[[var1, group_var]].dropna()
            groups = [
                group[var1].values for name, group in clean_data.groupby(group_var)
            ]
            groups = [g for g in groups if len(g) > 0]

            if len(groups) == 2:
                # Mann-Whitney U test
                try:
                    u_stat, p_value = stats.mannwhitneyu(
                        groups[0], groups[1], alternative="two-sided"
                    )

                    nonparam_results["mann_whitney_u"] = {
                        "statistic": u_stat,
                        "p_value": p_value,
                        "significant": p_value < alpha,
                    }

                    # Generate insights
                    if p_value < alpha:
                        group_names = clean_data[group_var].unique()
                        higher_group = (
                            group_names[0]
                            if np.median(groups[0]) > np.median(groups[1])
                            else group_names[1]
                        )
                        self.insights.append(
                            f"Significant difference in {var1} medians between {group_var} groups (Mann-Whitney U, {higher_group} higher)"
                        )

                except (ValueError, RuntimeWarning) as e:
                    nonparam_results["mann_whitney_error"] = str(e)

            elif len(groups) > 2:
                # Kruskal-Wallis test
                try:
                    h_stat, p_value = stats.kruskal(*groups)

                    nonparam_results["kruskal_wallis"] = {
                        "statistic": h_stat,
                        "p_value": p_value,
                        "significant": p_value < alpha,
                        "groups": len(groups),
                    }

                    # Generate insights
                    if p_value < alpha:
                        self.insights.append(
                            f"Significant difference in {var1} medians across {group_var} groups (Kruskal-Wallis test)"
                        )

                except (ValueError, RuntimeWarning) as e:
                    nonparam_results["kruskal_wallis_error"] = str(e)

        if nonparam_results:
            test_key = f"nonparam_{var1}"
            if var2:
                test_key += f"_vs_{var2}"
            elif group_var:
                test_key += f"_by_{group_var}"

            self.test_results[test_key] = nonparam_results

        return nonparam_results

    def comprehensive_statistical_testing(self):
        """Run comprehensive statistical testing suite"""
        print("📊 Performing comprehensive statistical testing...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        # Test normality for all numeric columns
        if numeric_cols:
            self.test_normality(numeric_cols)

        # Test correlations between numeric variables
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                self.correlation_tests(col1, col2)

        # Test associations between categorical variables
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i + 1 :]:
                self.chi_square_test(col1, col2)

        # ANOVA tests (numeric by categorical)
        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                # Only test if categorical variable has reasonable number of groups
                n_groups = self.df[cat_col].nunique()
                if 2 <= n_groups <= 10:
                    self.anova_test(num_col, cat_col)

        # Store insights
        self.test_results["statistical_insights"] = self.insights

        print("✅ Statistical testing completed!")
        return self.test_results
