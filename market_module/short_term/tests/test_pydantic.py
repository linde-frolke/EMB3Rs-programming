## Test pydantic 

from market_module.short_term.datastructures.inputstructs import MarketSettings

# good settings
settings = MarketSettings(nr_of_h=2, offer_type="simple", prod_diff="noPref", market_design="pool", network_type=None,
                    el_dependent=False, el_price=None)

# bad settings
settings = MarketSettings(nr_of_h=2, offer_type="simple", prod_diff="co2Emissions", market_design="pool", network_type=None,
                    el_dependent=False, el_price=None)