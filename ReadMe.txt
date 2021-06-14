# Prototype of Markets Module
The Market Module (MM) will provide the user with economic and fairness indicators including energy transaction, market price, social welfare, fairness among prices. This will be done by short-term and long market analyses that simulate various market structures and incorporate business conditions and network models. The MM will consider the existing Pool market as well as new, decentralized market setups, here named “community” and “peer-t-peer”. The modeling of heat/cold sources and sinks will include flexibility, offering price and business preferences. This document describes the implementation of the first prototype of the MM, which covers the programming of MM for the short-term analysis.

## Needed tools
The prototype is written in Python 3.8, and is organized using functions and classes written in .py files. Required Python packages are listed in the file ‘requirements.txt’.

## Main Features and access to the prototype
The first prototype contains only the short term market. It should also be noted that block bids have not been implemented yet. 

The first version of the MM prototype is structured as follows:
    • The inputs are divided over 3 categories: “settings”, “agent data”, and “network”. There is a class for each of these categories, which is used to create an object that contains all needed inputs for each category. These classes are defined in the file “datastructures/inputstructs.py”.
        ◦ Settings contains choices including number of time steps, offer type, market design, and whether product differentiation is used. 
        ◦ AgentData contains costs, utility, minimum and maximum generation and consumption for each agent and each hour. It also contains the agent_id for each agent. Optionally, it contains co2 emissions for each agent, and whether the agent is a part of the community.
        ◦ Network contains the distance between pairs of agents, and losses between pairs of agents. It may contain more in future versions.
    • For each of the market designs (Pool, P2P and Community) there is a separate function (in the folder “market_functions”) that creates the market optimization problem and solves it. The outcome is stored in an object of the “Result” class
    • The Result class (in datastructures/resultobject.py) is an object that contains values of primal and dual variables  that will be needed either 1) as output directly or 2) for the computation of other outputs. The Result object will also contain the output values that are computed from these optimized variables, such as the QoE, settlements, and social welfare.
        ◦ The result object has two methods. One for creating plots, and one for storing the entire result object in a .pickle file. 
    • The function named add_energy_budget() in file “market_functions/add_energy_budget.py” implements the energy budget constraint, that may be used in any of the 3 market types
    • For now, we have created some scripts with case studies for each market type to make sure the functions are working as they should. Until we are connected to some input data from user interface, we are using these scripts to test our module. They are located in the base folder and in “/test_setup_scripts”, with name ‘test_prototype_...’ and a description of the type of market that is tested.

The users can run the prototype through the py.file ‘test_prototype’ in the folder ‘test_setup_scripts/’ in the GitHub repo. Further instructions are also available in the README file.

## Inputs and Outputs
Inputs: (for now, inputs are written in scripts in the /test_setup_scripts/ folder. )
    • Settings: 
        ◦ number of time steps
        ◦ offer type
        ◦ market design
        ◦ type of product differentiation
    • Agent data:
        ◦ name/id (list of strings)
        ◦ type (producer/consumer/prosumer), one per agent
        ◦ cost (matrix of floats per hour per agent)
        ◦ utility (matrix of floats per hour per agent)
        ◦ min/max load and generation (matrices of floats per hour per agent)
        ◦ co2 emissions (list of floats per agent)
        ◦ part of community (list of boolean per agent)
    • Network:
        ◦ distance between pairs of nodes
        ◦ losses between pairs of nodes
        ◦ agent locations within the network

Outputs:
    • dispatch for each agent and hour
    • social welfare for each hour
    • market clearing price (also called shadow price) for each agent and hour
    • settlement for each agent (payments and revenues for each agent at each time)
    • fairness indicators: quality of experience for each hour
    • plot of market clearing for each hour

# Emb3rs project

EMB3Rs (“User-driven Energy-Matching & Business Prospection Tool for Industrial Excess Heat/Cold Reduction, Recovery and Redistribution) is a European project funded under the H2020 Programme (Grant Agreement N°847121) to develop an open-sourced tool to match potential sources of excess thermal energy with compatible users of heat and cold.

Users, like industries and other sources that produce excess heat, will provide the essential parameters, such as their location and the available excess thermal energy. The EMB3Rs platform will then autonomously and intuitively assess the feasibility of new business scenarios and identify the technical solutions to match these sources with compatible sinks. End users such as building managers, energy communities or individual consumers will be able to determine the costs and benefits of industrial excess heat and cold utilisation routes and define the requirements for implementing the most promising solutions. 

The EMB3Rs platform will integrate several analysis modules that will allow a full exploration of the feasible technical routes to the recovery and use of the available excess thermal energy. The code for each of the independent modules will be made available at the link to the EMB3Rs github repository.

# Authors
    • Sérgio Faria, developer
    • Linde Frölke, developer
    • Tiago Soares, supervision
    • Tiago Sousa, supervision







Conventions:
1. In matrix notation, rows iterate over time, columns iterate over agents. So Pmin[1,5] is time 1 agent 5.
2. Index for an agent is denoted by "i", index for time is "t", index for node is "n", for pipe is "p"
3. Power injection by an agent or node is POSITIVE when it is a net generator.
4. Trade is positive when energy is sold
5. If a 3-dim array would be needed, cvxpy cannot handle this. The solution we use is a list of matrix variables.
    For example, trades are needed for each hour, and agent pair. The list Tnm contains a matrix variable for each t
    So Tnm[t][i,j] is the trade from i to j at time t.
