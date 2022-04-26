## Test pydantic 

from market_module.short_term.datastructures.inputstructs import MarketSettings

# good settings
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", market_design="pool", network_type=None,
                    el_dependent=False, el_price=None)

# bad settings
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="co2Emissions", 
                market_design="pool", network_type=None,
                el_dependent=False, el_price=None)

# bad community settings 
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", 
                market_design="community", network_type=None,
                el_dependent=False, el_price=None, community_objective="autonomy", 
                                gamma_peak=10,
                                gamma_imp=None, 
                                gamma_exp=None)

settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", 
                market_design="community", network_type=None,
                el_dependent=False, el_price=None, community_objective="autonomy", 
                                gamma_peak=None,
                                gamma_imp=None, 
                                gamma_exp=None)

settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", 
                market_design="community", network_type=None,
                el_dependent=False, el_price=None, community_objective="autonomy", 
                                gamma_peak=5,
                                gamma_imp=None, 
                                gamma_exp=-10)
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", 
                market_design="community", network_type=None,
                el_dependent=False, el_price=None, community_objective="peakShaving", 
                                gamma_peak=None,
                                gamma_imp=5, 
                                gamma_exp=-6)
