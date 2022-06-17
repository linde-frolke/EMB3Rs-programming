import pandas as pd
import matplotlib.pyplot as plt, mpld3

def output_to_html(output_dict):
    df = pd.DataFrame(output_dict)
    df.index.name = "t"
    html = df.to_html(
        classes=['table'], 
        columns=list(output_dict.keys())
        )
    return html 

def output_to_plot(output_dict, ylab):
    df = pd.DataFrame(output_dict)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df.index, df, label=df.columns)
    ax.set_xlabel("hour number")
    ax.set_ylabel(ylab)
    ax.legend()
    # plt.tight_layout()
    fig_html = mpld3.fig_to_html(fig)

    return fig_html