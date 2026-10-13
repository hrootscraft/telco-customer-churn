"""Synthetic event log (bills, payments, usage, tickets, plan changes) behind the snapshot."""

from churnprop.events.simulate import EventLog, simulate_events

__all__ = ["EventLog", "simulate_events"]
