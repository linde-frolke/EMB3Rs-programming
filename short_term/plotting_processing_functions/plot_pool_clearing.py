import numpy as np


def prep_plot_market_clearing_pool(period, agent_data):
    """
    prepares some quantities for market clearing plot
    period is the period nr for which to plot
    agent_data is the relevant AgentData object
    """
    # Plotting
    hour = period  # hour to plot
    # Consumers
    data2 = []  # Sorted and cumulative producers heat offer
    yy_load = []  # Sorted producers offer
    # sorting according to the index (reverse)
    for i in np.argsort(agent_data.util.T[hour])[::-1]:
        data2.append(agent_data.lmax.T[hour][i])
        yy_load.append(agent_data.util.T[hour][i])
    data2 = np.cumsum(data2)
    data2 = np.insert(data2, 0, 0)
    data2 = np.insert(data2, -1, data2[-1])

    yy_load = np.insert(yy_load, 0, yy_load[0])
    yy_load = np.insert(yy_load, -1, yy_load[-1])

    # Producers
    data1 = []  # Sorted and cumulative producers heat offer
    yy_gen = []  # Sorted producers offer
    # sorting according to the index
    for i in np.argsort(agent_data.cost.T[hour]):
        data1.append(agent_data.gmax.T[hour][i])
        yy_gen.append(agent_data.cost.T[hour][i])
    data1 = np.cumsum(data1)
    data1 = np.insert(data1, 0, 0)
    data1 = np.insert(data1, -1, data1[-1])
    yy_gen = np.insert(yy_gen, 0, 0)
    yy_gen = np.insert(yy_gen, -1, yy_gen[-1])

    return data1, data2, yy_gen, yy_load
