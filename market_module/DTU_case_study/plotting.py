import unicodedata
import matplotlib.pyplot as plt
import pandas as pd
import unicodedata
import numpy as np

def plotting(results,model_name):
        #time_range = pd.date_range(start=f'{Year}-{Month}-{Day}',periods=nr_h,freq='H')
        uniform_price = results['shadow_price']['uniform price']
        Pn = pd.DataFrame.from_dict(results['Pn'])
        Gn = pd.DataFrame.from_dict(results['Gn'])
        Ln = pd.DataFrame.from_dict(results['Ln'])
    
        sw = pd.DataFrame.from_dict((results['social_welfare_h']['Social Welfare']))
        settlement = pd.DataFrame.from_dict(results['settlement'])
        shadow_price = pd.DataFrame.from_dict(results['shadow_price'])

        # Production and consumption of supermarket 
        fig, axes = plt.subplots()
        Ln.sm_1.plot(ax=axes)
        Gn.sm_1.plot(ax=axes)
        axes.set_title(f'Supermarket Consumption and Production ({model_name})')
        axes.legend(['Consumption','Production'])
        #axes.legend(['Consumption,Production'])
        axes.set_xlabel('Time[h]')
        axes.set_ylabel('kWh')
        axes.grid()

        # Production of grid and Market Price
        fig, axes = plt.subplots(2,1,sharex=True)
        axes[0].plot(Gn['grid_1'])
        axes[1].plot(shadow_price['uniform price'])
        axes[0].set_title(f'Grid Production ({model_name})')
        axes[1].set_title(f'Market Price ({model_name})')
        axes[1].set_xlabel('Time [h]')
        axes[0].grid()
        axes[1].grid()

        # Production of Grid and SuperMarket
        fig,axes = plt.subplots()
        Pn.grid_1.plot(ax=axes)
        Pn.sm_1.plot(ax=axes)
        axes.legend(['Grid','Supermarket'])
        axes.set_title(f'$P_n$ of Agents ({model_name})')
        axes.grid()

        # Social Welfare
        start = sw.min()[0]
        end = sw.max()[0]
        fig,axes = plt.subplots()
        sw.plot(ax=axes)
        axes.yaxis.set_ticks(np.arange(start,end,-(start-end)/10.0))
        axes.legend(['Social Welfare'],loc='upper left')
        axes.set_xlabel('Time [h]')
        axes.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
        axes.set_title(f'Social Welfare ({model_name})')
        axes.grid()

        # Settlement
        fig,axes = plt.subplots()
        settlement.grid_1.plot(ax=axes)
        settlement.sm_1.plot(ax=axes)
        axes.set_xlabel('Time [h]')
        axes.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
        axes.set_title(f'Settlement ({model_name})')
        axes.grid()
        axes.legend()
        plt.show()

        return 
