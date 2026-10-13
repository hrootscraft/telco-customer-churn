"""churnprop: retention propensity, offer targeting under a budget, and production ownership.

Built on the IBM Telco customer churn data (7,043 real subscribers). The package is
organised by the question each module answers:

* ``data`` / ``features``      load, clean, gate leakage, engineer and encode features
* ``propensity``               the honest 8-model benchmark, feature selection, calibration
* ``explain``                  SHAP drivers: global, per segment, per customer
* ``plots``                    every figure in ``reports/figures``
* ``cli``                      ``churnprop benchmark|explain ...``
"""

__version__ = "2.0.0"
