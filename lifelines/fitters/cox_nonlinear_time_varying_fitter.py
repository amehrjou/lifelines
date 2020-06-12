# -*- coding: utf-8 -*-


from datetime import datetime
import warnings
import time

import numpy as np
import pandas as pd
from scipy import stats

from numpy.linalg import norm, inv
from numpy import sum as array_sum_to_scalar
from scipy.linalg import solve as spsolve, LinAlgError
from autograd import elementwise_grad
from autograd import numpy as anp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim


from lifelines.fitters import SemiParametricRegressionFittter
from lifelines.fitters.mixins import ProportionalHazardMixin
from lifelines.utils.printer import Printer
from lifelines.statistics import _chisq_test_p_value, StatisticalResult
from lifelines.utils import (
    _get_index,
    _to_list,
    # check_for_overlapping_intervals,
    check_for_numeric_dtypes_or_raise,
    check_low_var,
    check_complete_separation_low_variance,
    check_for_immediate_deaths,
    check_for_instantaneous_events_at_time_zero,
    check_for_instantaneous_events_at_death_time,
    check_for_nonnegative_intervals,
    pass_for_numeric_dtypes_or_raise_array,
    ConvergenceError,
    ConvergenceWarning,
    inv_normal_cdf,
    normalize,
    StepSizer,
    check_nans_or_infs,
    string_justify,
    coalesce,
)


__all__ = ["CoxNonLinearTimeVaryingFitter"]

matrix_axis_0_sum_to_1d_array = lambda m: np.sum(m, 0)


class CoxNonLinearTimeVaryingFitter(SemiParametricRegressionFittter, ProportionalHazardMixin):
    r"""
    This class implements fitting Cox's nonlinear time-varying proportional hazard model:

        .. math::  h(t|x(t)) = h_0(t)\exp((x(t)-\overline{x})'\beta)

    Parameters
    ----------
    alpha: float, optional (default=0.05)
       the level in the confidence intervals.
    penalizer: float, optional
        the coefficient of an L2 penalizer in the regression

    Attributes
    ----------
    params_ : Series
        The estimated coefficients. Changed in version 0.22.0: use to be ``.hazards_``
    hazard_ratios_ : Series
        The exp(coefficients)
    confidence_intervals_ : DataFrame
        The lower and upper confidence intervals for the hazard coefficients
    event_observed: Series
        The event_observed variable provided
    weights: Series
        The event_observed variable provided
    variance_matrix_ : numpy array
        The variance matrix of the coefficients
    strata: list
        the strata provided
    standard_errors_: Series
        the standard errors of the estimates
    baseline_cumulative_hazard_: DataFrame
    baseline_survival_: DataFrame
    """

    _KNOWN_MODEL = True

    def __init__(self, alpha=0.05, penalizer=0.0, l1_ratio: float = 0.0, strata=None):
        super(CoxNonLinearTimeVaryingFitter, self).__init__(alpha=alpha)
        self.alpha = alpha
        self.penalizer = penalizer
        self.strata = strata
        self.l1_ratio = l1_ratio
        self.type_pt = torch.float

    def fit(
        self,
        df,
        event_col,
        start_col="start",
        stop_col="stop",
        weights_col=None,
        id_col=None,
        show_progress=False,
        step_size=None,
        epochs=1,
        batch_size=16,
        robust=False,
        strata=None,
        initial_point=None,
    ):  # pylint: disable=too-many-arguments
        """
        Fit the Cox Nonlinear Hazard model to a time varying dataset. Tied survival times
        are handled using Efron's tie-method.

        Parameters
        -----------
        df: DataFrame
            a Pandas DataFrame with necessary columns `duration_col` and
           `event_col`, plus other covariates. `duration_col` refers to
           the lifetimes of the subjects. `event_col` refers to whether
           the 'death' events was observed: 1 if observed, 0 else (censored).
        event_col: string
           the column in DataFrame that contains the subjects' death
           observation. If left as None, assume all individuals are non-censored.
        start_col: string
            the column that contains the start of a subject's time period.
        stop_col: string
            the column that contains the end of a subject's time period.
        weights_col: string, optional
            the column that contains (possibly time-varying) weight of each subject-period row.
        id_col: string, optional
            A subject could have multiple rows in the DataFrame. This column contains
           the unique identifier per subject. If not provided, it's up to the
           user to make sure that there are no violations.
        show_progress: since the fitter is iterative, show convergence
           diagnostics.
        robust: bool, optional (default: True)
            Compute the robust errors using the Huber sandwich estimator, aka Wei-Lin estimate. This does not handle
          ties, so if there are high number of ties, results may significantly differ. See
          "The Robust Inference for the Cox Proportional Hazards Model", Journal of the American Statistical Association, Vol. 84, No. 408 (Dec., 1989), pp. 1074- 1078
        step_size: float, optional
            set an initial step size for the fitting algorithm.
        strata: list or string, optional
            specify a column or list of columns n to use in stratification. This is useful if a
            categorical covariate does not obey the proportional hazard assumption. This
            is used similar to the `strata` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.
        initial_point: (d,) numpy array, optional
            initialize the starting point of the iterative
            algorithm. Default is the zero vector.

        Returns
        --------
        self: CoxNonLinearTimeVaryingFitter
            self, with additional properties like ``hazards_`` and ``print_summary``

        """
        self.strata = coalesce(strata, self.strata)
        self.robust = robust
        if self.robust:
            raise NotImplementedError("Not available yet.")

        self.event_col = event_col
        self.id_col = id_col
        self.stop_col = stop_col
        self.start_col = start_col
        self._time_fit_was_called = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S") + " UTC"

        df = df.copy()

        if not (event_col in df and start_col in df and stop_col in df):
            raise KeyError("A column specified in the call to `fit` does not exist in the DataFrame provided.")

        if weights_col is None:
            self.weights_col = None
            assert "__weights" not in df.columns, "__weights is an internal lifelines column, please rename your column first."
            df["__weights"] = 1.0
        else:
            self.weights_col = weights_col
            if (df[weights_col] <= 0).any():
                raise ValueError("values in weights_col must be positive.")

        df = df.rename(columns={event_col: "event", start_col: "start", stop_col: "stop", weights_col: "__weights"})

        if self.strata is not None and self.id_col is not None:
            df = df.set_index(_to_list(self.strata) + [id_col])
            df = df.sort_index()
        elif self.strata is not None and self.id_col is None:
            df = df.set_index(_to_list(self.strata))
        elif self.strata is None and self.id_col is not None:
            df = df.set_index([id_col])

        events, start, stop = (
            pass_for_numeric_dtypes_or_raise_array(df.pop("event")).astype(bool),
            df.pop("start"),
            df.pop("stop"),
        )
        weights = df.pop("__weights").astype(float)

        df = df.astype(float)
        self._check_values(df, events, start, stop)

        self._norm_mean = df.mean(0)
        self._norm_std = df.std(0)

        # params_ = self._newton_rhaphson(
        #     normalize(df, self._norm_mean, self._norm_std),
        #     events,
        #     start,
        #     stop,
        #     weights,
        #     initial_point=initial_point,
        #     show_progress=show_progress,
        #     step_size=step_size,
        # )

        # Network architecture
        # in_features = d
        # hidden_features = 16
        # out_features = 1
        # batch_norm = True
        # dropout = 0.1
        # type_pt = torch.float

        initial_net = self.net

        self.net = self._neural_cox(
            normalize(df, self._norm_mean, self._norm_std),
            events,
            start,
            stop,
            weights,
            net=initial_net,
            show_progress=show_progress,
            training_epochs=epochs,
            batch_size=batch_size,
            step_size=step_size,
        )

        self.beta_params_ = pd.Series(list(self.net.beta.parameters())[0].detach().numpy().ravel(), name="coef")
        # self.variance_matrix_ = pd.DataFrame(-inv(self._hessian_) / np.outer(self._norm_std, self._norm_std), index=df.columns)
        # self.standard_errors_ = self._compute_standard_errors(base
        #     normalize(df, self._norm_mean, self._norm_std), events, start, stop, weights
        # )
        # self.confidence_intervals_ = self._compute_confidence_intervals()
        self.baseline_cumulative_hazard_ = self._compute_cumulative_baseline_hazard(df, events, start, stop, weights)
        self.baseline_survival_ = self._compute_baseline_survival()
        self.event_observed = events
        self.start_stop_and_events = pd.DataFrame({"event": events, "start": start, "stop": stop})
        self.weights = weights
        self._n_examples = df.shape[0]
        self._n_unique = df.index.unique().shape[0]
        return self

    def _neural_cox(
        self, X, events, start, stop, weights, net, show_progress=True, training_epochs=10, batch_size=16, step_size=0.01
    ):

        events = events.values.reshape(-1, 1)
        start = start.values.reshape(-1, 1)
        stop = stop.values.reshape(-1, 1)
        weights = weights.values.reshape(-1, 1)

        n, d = X.shape
        # create your optimizer
        optimizer = optim.Adam(net.parameters(), lr=step_size, weight_decay=0.01)
        start_time = time.time()

        full_table = np.concatenate([X, events, start, stop, weights], axis=1)

        loader = DataLoader(
            full_table,
            batch_size=batch_size,
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=None,
            pin_memory=False,
            drop_last=True,
            timeout=0,
            worker_init_fn=None,
        )

        for epoch in range(training_epochs):
            for batch_ndx, batch_data in enumerate(loader):
                X, events, start, stop, weights = (
                    batch_data[:, 0:d],
                    batch_data[:, d],
                    batch_data[:, d + 1],
                    batch_data[:, d + 2],
                    batch_data[:, d + 3],
                )
                X = X.to(self.type_pt)
                weights = weights.to(self.type_pt)
                events = events.to(torch.bool)
                start = start.to(torch.int)
                stop = stop.to(torch.int)
                optimizer.zero_grad()  # zero the gradient buffers
                batch_negloglik = -self._get_log_lik(X, events, start, stop, weights, net)
                batch_negloglik.backward()
                optimizer.step()
                print(batch_negloglik)
        return net

    def create_net(self, in_features, hidden_features, out_features, batch_norm, dropout):
        self.net = Net(in_features, hidden_features, out_features, batch_norm, dropout, self.type_pt)

    def _check_values(self, df, events, start, stop):
        # check_for_overlapping_intervals(df) # this is currently too slow for production.
        check_nans_or_infs(df)
        check_low_var(df)
        check_complete_separation_low_variance(df, events, self.event_col)
        check_for_numeric_dtypes_or_raise(df)
        check_for_nonnegative_intervals(start, stop)
        check_for_immediate_deaths(events, start, stop)
        check_for_instantaneous_events_at_time_zero(start, stop)
        check_for_instantaneous_events_at_death_time(events, start, stop)

    def _partition_by_strata(self, X, events, start, stop, weights):
        for stratum, stratified_X in X.groupby(self.strata):
            stratified_W = weights.loc[stratum]
            stratified_start = start.loc[stratum]
            stratified_events = events.loc[stratum]
            stratified_stop = stop.loc[stratum]
            yield (
                stratified_X.values,
                stratified_events.values,
                stratified_start.values,
                stratified_stop.values,
                stratified_W.values,
            ), stratum

    def _partition_by_strata_and_apply(self, X, events, start, stop, weights, function, *args):
        for ((stratified_X, stratified_events, stratified_start, stratified_stop, stratified_W), _) in self._partition_by_strata(
            X, events, start, stop, weights
        ):
            yield function(stratified_X, stratified_events, stratified_start, stratified_stop, stratified_W, *args)

    def _compute_z_values(self):
        # return self.params_ / self.standard_errors_
        raise NotImplementedError

    def _compute_p_values(self):
        # U = self._compute_z_values() ** 2
        # return stats.chi2.sf(U, 1)
        raise NotImplementedError

    def _compute_confidence_intervals(self):
        # ci = 100 * (1 - self.alpha)
        # z = inv_normal_cdf(1 - self.alpha / 2)
        # se = self.standard_errors_
        # hazards = self.params_.values
        # return pd.DataFrame(
        #     np.c_[hazards - z * se, hazards + z * se],
        #     columns=["%g%% lower-bound" % ci, "%g%% upper-bound" % ci],
        #     index=self.params_.index,
        # )
        raise NotImplementedError

    @property
    def summary(self):
        """Summary statistics describing the fit.

        Returns
        -------
        df : DataFrame
            Contains columns coef, np.exp(coef), se(coef), z, p, lower, upper"""
        # ci = 100 * (1 - self.alpha)
        # z = inv_normal_cdf(1 - self.alpha / 2)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
            df = pd.DataFrame(index=self.params_.index)
            df["last layer params"] = self.beta_params_
            df["exp(coef)"] = self.hazard_ratios_
            # df["se(coef)"] = self.standard_errors_
            # df["coef lower %g%%" % ci] = self.confidence_intervals_["%g%% lower-bound" % ci]
            # df["coef upper %g%%" % ci] = self.confidence_intervals_["%g%% upper-bound" % ci]
            # df["exp(coef) lower %g%%" % ci] = self.hazard_ratios_ * np.exp(-z * self.standard_errors_)
            # df["exp(coef) upper %g%%" % ci] = self.hazard_ratios_ * np.exp(z * self.standard_errors_)
            # df["z"] = self._compute_z_values()
            # df["p"] = self._compute_p_values()
            # df["-log2(p)"] = -np.log2(df["p"])
            return df

    # def _neural_cox(
    #     self,
    #     df,
    #     events,
    #     start,
    #     stop,
    #     weights,
    #     show_progress=False,
    #     step_size=None,
    #     precision=10e-6,
    #     max_steps=50,
    #     initial_net=None,
    # ):  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    #     """
    #     Nonlinear model (neural net) for fitting CPH model.

    #     Parameters
    #     ----------
    #     df: DataFrame
    #     stop_times_events: DataFrame
    #         meta information about the subjects history
    #     show_progress: bool, optional (default: True)
    #         to show verbose output of convergence
    #     step_size: float
    #         > 0 to determine a starting step size in NR algorithm.
    #     precision: float
    #         the convergence halts if the norm of delta between
    #                 successive positions is less than epsilon.

    #     Returns
    #     --------
    #     net: (1,d) numpy array.
    #     """

    #     raise NotImplementedError

    def _get_log_lik(self, X_pt, events_pt, start_pt, stop_pt, weights_pt, net):
        """
        Calculate the pytorch compatibale log-likelihood
        -------
        X: (m, d) tensor of covariates,
        events: (1, d) tensor of events
        start: (1, d) tensor of start times
        stop: (1, d) tensor of stop times
        weights: (1, d) tensor of weight times
        net: the current state of nonlinear link function h(t|x) = h_0(t|x)exp(net(x))
        """
        log_lik = 0
        events_pt = events_pt.to(torch.bool)
        unique_death_times = np.unique(stop_pt[events_pt])
        for t in unique_death_times:
            ix = (start_pt < t) & (t <= stop_pt)
            X_at_t_pt = X_pt[ix]
            weights_at_t_pt = weights_pt[ix][:, None]
            stops_events_at_t_pt = stop_pt[ix]
            events_at_t_pt = events_pt[ix]

            #             X_at_t_pt = torch.tensor(X_at_t.values, dtype=type_pt)
            #             weights_at_t_pt = torch.tensor(weights_at_t, dtype=type_pt)
            #             stops_events_at_t_pt = torch.tensor(stops_events_at_t, dtype=type_pt)
            #             events_at_t_pt = torch.tensor(events_at_t, dtype=type_pt)

            phi_i = weights_at_t_pt * torch.exp(net(X_at_t_pt))
            risk_phi = torch.sum(phi_i, dim=0)  # Calculate sums of Risk set

            # Calculate the sums of Tie set
            #             print(events_at_t)
            #             print( (stops_events_at_t == t))
            #             print(deaths)
            #             print(events_at_t.detach().numpy().astype(int))
            #             print(stops_events_at_t.detach().numpy() == t)
            deaths_pt = events_at_t_pt & (stops_events_at_t_pt == t)
            deaths = deaths_pt.detach().numpy()
            ties_counts = array_sum_to_scalar(deaths)  # should always at least 1. Why? TODO
            xi_deaths = X_at_t_pt[deaths]
            weights_deaths = weights_at_t_pt[deaths]
            phi_death_sum = torch.sum(phi_i[deaths], dim=0)
            weight_count = torch.sum(weights_at_t_pt[deaths], axis=0)
            weighted_average = weight_count / ties_counts

            # No tie
            for l in range(ties_counts):
                if ties_counts > 1:
                    increasing_proportion = l / ties_counts
                    denom = risk_phi - increasing_proportion * phi_death_sum
                else:
                    denom = risk_phi
                log_lik -= weighted_average * torch.log(denom)
            log_lik += phi_death_sum
        return log_lik

    def predict_log_partial_hazard(self, X) -> pd.Series:
        r"""
        This is equivalent to R's linear.predictors.
        Returns the log of the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`(x - \bar{x})'\beta`


        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        DataFrame

        Note
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.
        """
        if isinstance(X, pd.DataFrame):
            # order = self.params_.index
            # X = X[order]
            check_for_numeric_dtypes_or_raise(X)

        X = X.astype(float)
        X = normalize(X, self._norm_mean.values, 1)

        return pd.Series(self.net(torch.tensor(X.values, dtype=self.type_pt)).detach().numpy().ravel())

    def predict_partial_hazard(self, X) -> pd.Series:
        r"""
        Returns the partial hazard for the individuals, partial since the
        baseline hazard is not included. Equal to :math:`\exp{(x - \bar{x})'\beta }`

        Parameters
        ----------
        X: numpy array or DataFrame
            a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns
        -------
        DataFrame

        Note
        -----
        If X is a DataFrame, the order of the columns do not matter. But
        if X is an array, then the column ordering is assumed to be the
        same as the training dataset.

        """
        return np.exp(self.predict_log_partial_hazard(X))

    def print_summary(self, decimals=2, style=None, **kwargs):
        """
        Print summary statistics describing the fit, the coefficients, and the error bounds.

        Parameters
        -----------
        decimals: int, optional (default=2)
            specify the number of decimal places to show
        style: string
            {html, ascii, latex}
        kwargs:
            print additional meta data in the output (useful to provide model names, dataset names, etc.) when comparing
            multiple outputs.

        """
        justify = string_justify(18)

        headers = []

        if self.event_col:
            headers.append(("event col", "'%s'" % self.event_col))
        if self.weights_col:
            headers.append(("weights col", "'%s'" % self.weights_col))
        if isinstance(self.penalizer, np.ndarray) or self.penalizer > 0:
            headers.append(("penalizer", self.penalizer))
        if self.strata:
            headers.append(("strata", self.strata))

        headers.extend(
            [
                ("number of subjects", self._n_unique),
                ("number of periods", self._n_examples),
                ("number of events", self.event_observed.sum()),
                ("partial log-likelihood", "{:.{prec}f}".format(self.log_likelihood_, prec=decimals)),
                ("time fit was run", self._time_fit_was_called),
            ]
        )

        sr = self.log_likelihood_ratio_test()
        footers = []
        footers.extend(
            [
                ("Partial AIC", "{:.{prec}f}".format(self.AIC_partial_, prec=decimals)),
                (
                    "log-likelihood ratio test",
                    "{:.{prec}f} on {} df".format(sr.test_statistic, sr.degrees_freedom, prec=decimals),
                ),
                ("-log2(p) of ll-ratio test", "{:.{prec}f}".format(-np.log2(sr.p_value), prec=decimals)),
            ]
        )

        p = Printer(self, headers, footers, justify, decimals, kwargs)
        p.print(style=style)

    def log_likelihood_ratio_test(self):
        # """
        # This function computes the likelihood ratio test for the Cox model. We
        # compare the existing model (with all the covariates) to the trivial model
        # of no covariates.

        # Conveniently, we can actually use CoxPHFitter class to do most of the work.

        # """
        # if hasattr(self, "_log_likelihood_null"):
        #     ll_null = self._log_likelihood_null

        # else:
        #     trivial_dataset = self.start_stop_and_events
        #     trivial_dataset = trivial_dataset.join(self.weights)
        #     trivial_dataset = trivial_dataset.reset_index()
        #     print("trivial dataset")
        #     print(trivial_dataset)
        #     ll_null = (
        #         CoxNonLinearTimeVaryingFitter()
        #         .fit(
        #             trivial_dataset,
        #             start_col=self.start_col,
        #             stop_col=self.stop_col,
        #             event_col=self.event_col,
        #             id_col=self.id_col,
        #             weights_col="__weights",
        #             strata=self.strata,
        #         )
        #         .log_likelihood_
        #     )

        # ll_alt = self.log_likelihood_
        # test_stat = 2 * (ll_alt - ll_null)
        # degrees_freedom = self.params_.shape[0]
        # p_value = _chisq_test_p_value(test_stat, degrees_freedom=degrees_freedom)
        # return StatisticalResult(
        #     p_value, test_stat, name="log-likelihood ratio test", degrees_freedom=degrees_freedom, null_distribution="chi squared"
        # )
        raise NotImplementedError

    # def plot(self, columns=None, ax=None, **errorbar_kwargs):
    #     """
    #     Produces a visual representation of the coefficients, including their standard errors and magnitudes.

    #     Parameters
    #     ----------
    #     columns : list, optional
    #         specify a subset of the columns to plot
    #     errorbar_kwargs:
    #         pass in additional plotting commands to matplotlib errorbar command

    #     Returns
    #     -------
    #     ax: matplotlib axis
    #         the matplotlib axis that be edited.

    #     """
    #     from matplotlib import pyplot as plt

    #     if ax is None:
    #         ax = plt.gca()

    #     errorbar_kwargs.setdefault("c", "k")
    #     errorbar_kwargs.setdefault("fmt", "s")
    #     errorbar_kwargs.setdefault("markerfacecolor", "white")
    #     errorbar_kwargs.setdefault("markeredgewidth", 1.25)
    #     errorbar_kwargs.setdefault("elinewidth", 1.25)
    #     errorbar_kwargs.setdefault("capsize", 3)

    #     z = inv_normal_cdf(1 - self.alpha / 2)

    #     if columns is None:
    #         user_supplied_columns = False
    #         columns = self.params_.index
    #     else:
    #         user_supplied_columns = True

    #     yaxis_locations = list(range(len(columns)))
    #     symmetric_errors = z * self.standard_errors_[columns].values.copy()
    #     hazards = self.params_[columns].values.copy()

    #     order = list(range(len(columns) - 1, -1, -1)) if user_supplied_columns else np.argsort(hazards)

    #     ax.errorbar(hazards[order], yaxis_locations, xerr=symmetric_errors[order], **errorbar_kwargs)
    #     best_ylim = ax.get_ylim()
    #     ax.vlines(0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65)
    #     ax.set_ylim(best_ylim)

    #     tick_labels = [columns[i] for i in order]

    #     ax.set_yticks(yaxis_locations)
    #     ax.set_yticklabels(tick_labels)
    #     ax.set_xlabel("log(HR) (%g%% CI)" % ((1 - self.alpha) * 100))

    #     return ax

    def _compute_cumulative_baseline_hazard(self, tv_data, events, start, stop, weights):  # pylint: disable=too-many-locals

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hazards = self.predict_partial_hazard(tv_data).values

        unique_death_times = np.unique(stop[events.values])
        baseline_hazard_ = pd.DataFrame(np.zeros_like(unique_death_times), index=unique_death_times, columns=["baseline hazard"])

        for t in unique_death_times:
            ix = (start.values < t) & (t <= stop.values)

            events_at_t = events.values[ix]
            stops_at_t = stop.values[ix]
            weights_at_t = weights.values[ix]
            hazards_at_t = hazards[ix]

            deaths = events_at_t & (stops_at_t == t)

            death_counts = (weights_at_t.squeeze() * deaths).sum()  # should always be atleast 1.
            baseline_hazard_.loc[t] = death_counts / hazards_at_t.sum()

        return baseline_hazard_.cumsum()

    def _compute_baseline_survival(self):
        survival_df = np.exp(-self.baseline_cumulative_hazard_)
        survival_df.columns = ["baseline survival"]
        return survival_df

    def __repr__(self):
        classname = self._class_name
        try:
            s = """<lifelines.%s: fitted with %d periods, %d subjects, %d events>""" % (
                classname,
                self._n_examples,
                self._n_unique,
                self.event_observed.sum(),
            )
        except AttributeError:
            s = """<lifelines.%s>""" % classname
        return s

    def _compute_residuals(self, df, events, start, stop, weights):
        raise NotImplementedError()

    # def _compute_delta_beta(self, df, events, start, stop, weights):
    #         """ approximate change in betas as a result of excluding ith row"""

    #         score_residuals = self._compute_residuals(df, events, start, stop, weights) * weights[:, None]

    #         naive_var = inv(self._hessian_)
    #         delta_betas = -score_residuals.dot(naive_var) / self._norm_std.values

    #         return delta_betas

    # def _compute_sandwich_estimator(self, X, events, start, stop, weights):

    #     delta_betas = self._compute_delta_beta(X, events, start, stop, weights)

    #     if self.cluster_col:
    #         delta_betas = pd.DataFrame(delta_betas).groupby(self._clusters).sum().values

    #     sandwich_estimator = delta_betas.T.dot(delta_betas)
    #     return sandwich_estimator

    # def _compute_standard_errors(self, X, events, start, stop, weights):
    #     if self.robust:
    #         se = np.sqrt(self._compute_sandwich_estimator(X, events, start, stop, weights).diagonal())
    #     else:
    #         se = np.sqrt(self.variance_matrix_.values.diagonal())
    #     return pd.Series(se, index=self.params_.index, name="se")


class Net(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, batch_norm=False, dropout=0.0, type_pt=torch.float):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False).to(type_pt)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        # self.dp1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.beta = nn.Linear(hidden_features, out_features, bias=False).to(type_pt)
        torch.nn.init.xavier_uniform(self.beta.weight)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.beta(x)
        return x
