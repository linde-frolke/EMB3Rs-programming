
Welcome to EMB3Rs's Market Module documentation!
================================================

The EMB3Rs platform has been designed to evaluate the reuse and trade of waste Heating
and Cooling (HC) in a holistic perspective within an industrial process,
energy system environment, or in District Heating and Cooling (DHC) systems
under-regulated and liberalized market environment. The platform empowers
industrial users and stakeholders to investigate the economic potential
of their investment in the recovery of waste HC as an energy resource, based on
the simulation of supply-demand scenarios. To this end, the platform is able to
simulate multiple business and market models for DHC systems.
The EMB3Rs platform is composed of five modules: Core Functionalities (CF),
Geographical Information System (GIS), Techno-Economic Optimisation (TEO),
Market Module (MM), and Business Module (BM).
The dedicated MM allows users to simulate current and future trends for the HC markets,
allowing them to choose the best market framework aligned with the users’ economic,
environmental and social interests. This is especially important for users who have
invested (or are considering investing) in waste heat recovery to assess the economic
potential and environmental savings of their investment. Recently, peer-to-peer (P2P)
and community-based markets have been presented as an alternative to existing pool
energy markets, both for HC and electricity systems. Therefore, the MM models and
implements the P2P and community-based market designs, in addition to the conventional
pool market design. In this way, users can create, test and validate different market
structures for selling and buying energy in DHC systems. The outputs of the market
analysis enable users (e.g., industries, supermarkets and data centres) to estimate
potential costs and revenues for different market participants from trading excess
heat and cold, under different market designs.
Based on a user’s experience and motivation, different types of market analysis might
be required. To address this, the MM was divided into short-term and long-term analyses.
The short-term market analysis is intended to give an overview of individual agents’
performance when subjected to different market features. In this design, users are
able to change some parameters and easily and quickly check the effect of such changes
on market outcomes. In this way, the effect of different market settings can be studied
in case studies with short horizons. On the other hand, the long-term market analysis
can be used to investigate market outcomes for agents in wider time ranges like months
or years. This is more suitable when the user wants to evaluate the profitability of
potential investments.



Short-Term
==================================

Inputs
~~~~~~
.. list-table::
   :widths: 7 25
   :header-rows: 1

   * - Name
     - Description

   * - md
     - Market design. Represents the market design the user intends to simulate. Pool, P2P and Community are the options.

   * - nr_of_hours
     - Represents the simulation horizon period. An integer is expected.

   * - start_datetime
     - Date of the format “dd-mm-yyyy”.

   * - offer_type
     - Represents the offer type, which can be related to all agents or only to some agents. Simple, Block and energyBudget are the options

   * - prod_diff
     - The option related to product differentiation, namely, by encouraging or penalizing some market transactions, according to user preferences. noPref (No Preferences), co2Emissions (CO2 Emissions), networkDistance (Network Distance) and losses (Energy Losses)   are the options  .

   * - el_dependent
     - Only available for Pool market True or False.

   * - el_price
     - If el_dependent=True, an electricity price (forecast) must be entered for all market hours.

   * - Network
     - "none" or "direction".

   * - Objective
     - Only if “md = community”. Should be one of [“autonomy”, “peakShaving”].

   * - Community settings
     - Include g_peak, g_exp, g_imp. g_exp must be nonpositive, while g_imp and g_peak must be nonnegative.

Parameters
~~~~~~~~~~

.. list-table::
   :widths: 7 25
   :header-rows: 1

   * - Name
     - Description

   * - agent_ids [n]
     - Represents each agent id. This is imported from TEO.

   * - co2_emissions [n]
     - CO2 emissions by agent, imported from TEO. This list is only required when preferences on CO2-Emissions are selected in the P2P market.

   * - gmax [t,n]
     - The maximum production each agent offers in the market. Values must be provided for each agent and each time step. A constant value for each agent is imported from the TEO module, but an optional user input can override the imported values.

   * - lmax [t,n]
     - Maximum consumption each agent offers to purchase in the market. Values must be provided for each agent and each time step. This load profile is imported from the TEO module.

   * - cost [t,n]
     - The offer price is related to the production, which represents the minimum price the agent wants to receive per unit of energy. Values must be provided for each agent and each time step. This is imported from the TEO module.

   * - util [t,n]
     - This bid is related to consumption. Values must be provided for each agent and each time step.

   * - gis_data
     - All the network data that is required to run the MM under the Distance or Losses product differentiation features. It must be a dictionary with the linked agents, the total length between them, and the total costs associated with each pipeline. This is also used in case the network feature is selected in the Pool market. It is imported from the GIS module.

   * - block_offer
     - A dictionary with agent IDs as keys. The values include the time steps when the block offer is active. Not all agent IDs need to be included, only the IDs of agents that submit block bids are needed.

   * - is_in_community
     - A boolean for each agent specifies whether it is part of the community or not. This input is required when the Community market is chosen.

   * - is_chp
     - A boolean for each agent specifies whether it is a CHP or not. This input is mandatory in case the user selects the electricity dependence option. This input is derived from agent IDs provided by the TEO, so the user does not need to input this.

   * - chp_pars
     - For each agent that is a CHP, some parameters must be specified. This input is mandatory in case the user selects the electricity dependence option, otherwise, it is not needed.

Variables
~~~~~~~~~

.. list-table::
   :widths: 7 25
   :header-rows: 1

   * - Name
     - Description

   * - Ln [t,n]
     - Amount purchased in the market for each agent for each time step.

   * - Gn [t,n]
     - Amount sold in the market for each agent for each time step.

   * - Pn [t,n]
     - Represents the net balance for each agent for each time step.

   * - Snm [t,n,n]
     - Amount sold by agent n to agent m, for each agent, for each time step. This variable is only used in the decentralized market.

   * - Bnm [t,n,n]
     - Amount bought by agent n to agent m, for each agent, for each time step. This variable is only used in the decentralized market.

   * - Tnm [t,n,n]
     - Represents the net balance for each bilateral trade, for each agent, for each time step.

   * - b [t,n]
     - Binary variable indicating whether a bid is fully accepted or fully rejected. This variable is only used if the block offer is selected.

   * - shadow_price [t,n,n] or [t,n]
     - Presents the market clearing price. Presents one value per time step, if the centralized market design is chosen. Outputs one value per transaction and per time step, if the decentralized market design is the selected one.

Outputs
~~~~~~~

.. list-table::
   :widths: 7 25
   :header-rows: 1

   * - Name
     - Description

   * - Ln [t,n]
     - Amount purchased in the market for each agent for each time step.

   * - Gn [t,n]
     - Amount sold in the market for each agent for each time step.

   * - Pn [t,n]
     - Represents the net balance for each agent for each time step.

   * - Settlement [t,n]
     - The settlement is obtained by multiplying the energy dispatched by the price of each transaction. It is calculated for each agent for each time step.

   * - social_welfare [t]
     - The social welfare is obtained by multiplying the energy dispatched by the bid of each agent and then grouping the results by time step. A value is presented for each time step.

   * - shadow_price [t,n,n] or [t,n]
     - Represents the market clearing price. It presents one value per time step if the centralized market design is selected. Outputs one value per transaction and per time step, if the decentralized market design is selected.

   * - Market plot
     - Yields a plot with the offers’ merit order for all agents, for a single selected time step  . It is only available if the Pool market is the simulated design.

   * - QoE
     - Indicates the fairness level for each market result. The closer this indicator is to 1, the fairer the results will be to consumers. Outputs one value per time step. This value is only available in the P2P market design.

Long-Term
==================================

Inputs
~~~~~~
.. list-table::
   :widths: 7 25
   :header-rows: 1

   * - Name
     - Description

   * - md
     - Represents the market design the user intends to simulate. Centralized or Decentralized are the options.

   * - horizon_basis
     - Represents the simulation horizon period. Weeks, Months or Years are the options.

   * - data_profile
     - Represents the level of data aggregation, which can be considered as hourly or daily grouped. That is, it sets whether the optimization process is simulated on an hourly or daily basis for the entire time horizon. Note that this option influences the computational effort of the MM. Hourly or Daily are the options.

   * - recurrence
     - Represents the number of periods selected in the horizon_basis. An integer is expected.

   * - start_datetime
     - A data of format “dd-mm-yyyy”.

   * - yearly_demand_rate
     - The expected yearly demand rate change. The demand can increase or decrease over the years so a float number within the range [-1,1] is expected.

   * - prod_diff_option
     - The option is related to product differentiation, namely, by encouraging or penalizing some market transactions, according to user preferences. noPref (No Preference), co2Emissions (CO2 Emissions) and networkDistance (Network Distance) are the options.

Parameters
~~~~~~~~~~

.. list-table::
   :widths: 7 25
   :header-rows: 1

   * - Name
     - Description

   * - agent_ids [n]
     - Represents each agent id. This is imported from TEO.

   * - co2_emissions [n]
     - CO2 emissions by agent, imported from TEO. This list is only required when preferences on CO2-Emissions are selected in the decentralized model.

   * - gmax [t,n]
     - The maximum production each agent offers in the market. Values must be provided for each agent and each time step. A constant value for each agent is imported from the TEO module, but an optional user input can override the imported values.

   * - lmax [t,n]
     - Maximum consumption each agent offers to purchase in the market. Values must be provided for each agent and each time step. This load profile is imported from the TEO module.

   * - cost [t,n]
     - The offer price is related to the production, which represents the minimum price the agent wants to receive per unit of energy. Values must be provided for each agent and each time step. This is imported from the TEO module.

   * - util [t,n]
     - This bid is related to consumption. Values must be provided for each agent and each time step.

   * - gis_data
     - The network data to run the platform under the distance product differentiation feature. It must be a dictionary with the linked agents, the total length between them and the total cost associated with each pipeline. Such information is imported from the GIS module.


Variables
~~~~~~~~~

.. list-table::
   :widths: 7 25
   :header-rows: 1

   * - Name
     - Description

   * - Ln [t,n]
     - Amount purchased in the market for each agent for each time step.

   * - Gn [t,n]
     - Amount sold in the market for each agent for each time step.

   * - Pn [t,n]
     - Represents the net balance for each agent for each time step.

   * - Snm [t,n,n]
     - Amount sold by agent n to agent m, for each agent, for each time step. This variable is only used in the decentralized market.

   * - Bnm [t,n,n]
     - Amount bought by agent n to agent m, for each agent, for each time step. This variable is only used in the decentralized market.

   * - Tnm [t,n,n]
     - Represents the net balance for each bilateral trade, for each agent, for each time step.

   * - shadow_price [t,n,n] or [t,n]
     - Presents the market clearing price. Presents one value per time step, if the centralized market design is chosen. Outputs one value per transaction and per time step, if the decentralized market design is the selected one.

Outputs
~~~~~~~

.. list-table::
   :widths: 7 25
   :header-rows: 1

   * - Name
     - Description

   * - Ln [t,n]
     - Amount purchased in the market for each agent for each time step.

   * - Gn [t,n]
     - Amount sold in the market for each agent for each time step.

   * - Pn [t,n]
     - Represents the net balance for each agent for each time step.

   * - Settlement [t,n]
     - The settlement is obtained by multiplying the energy dispatched by the price of each transaction. It is calculated for each agent for each time step.

   * - agent_operational_cost [t,n]
     - The agent operating cost is obtained by multiplying the energy dispatched by the bid of each agent. It is calculated for each agent for each time step.

   * - social_welfare [t]
     - The social welfare is obtained by multiplying the energy dispatched by the bid of each agent and then grouping the results by time step. A value is presented for each time step.

   * - shadow_price [t,n,n] or [t,n]
     - Represents the market clearing price. It presents one value per time step if the centralized market design is selected. Outputs one value per transaction and per time step, if the decentralized market design is selected.

   * - SPM
     - This KPI indicates the percentage of successful participation in the market by sources and sinks. One value is presented per agent.

   * - ADG
     - This KPI indicates the average dispatched production by a source. The dispatched production by period is based on the ratio between the available capacity and the actual dispatched production.

   * - expensive_prod
     - Indicates the best price an agent must offer in the market to achieve higher revenue. The output will be one value since one agent and time step must be selected.

   * - QoE
     - Indicates the fairness level for each market result. The closer this indicator is to 1, the fairer the results will be. Outputs one value per time step. This value is only available in the decentralized market design.
