"""Offers: the menu, the planted customer responses, and the potential outcomes they imply."""

from churnprop.offers.menu import Offer, OffersConfig, load_offers
from churnprop.offers.simulate import PotentialOutcomes, plant_responses

__all__ = ["Offer", "OffersConfig", "PotentialOutcomes", "load_offers", "plant_responses"]
