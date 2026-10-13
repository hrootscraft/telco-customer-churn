"""Multi-offer uplift: the learner, budget allocation, and off-policy evaluation."""

from churnprop.uplift.allocate import allocate_budget, lp_allocation
from churnprop.uplift.learner import MultiArmUplift
from churnprop.uplift.ope import dr_value, ips_value

__all__ = ["MultiArmUplift", "allocate_budget", "dr_value", "ips_value", "lp_allocation"]
