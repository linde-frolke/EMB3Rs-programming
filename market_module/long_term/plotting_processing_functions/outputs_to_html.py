import pandas as pd
import matplotlib.pyplot as plt, mpld3

def output_to_html(output_dict):
    df = pd.DataFrame(output_dict)
    df.index.name = "t"
    html = df.round(decimals=2).to_html(
        classes=['table'], 
        columns=list(output_dict.keys())
        )
    return html

def output_to_html_no_index(output_dict):
    df = pd.DataFrame([output_dict])
    html = df.round(decimals=2).to_html(
        classes=['table'],
        columns=(output_dict.keys())
        )
    return html

def output_to_html_no_index_transpose(output_dict):
    df = pd.DataFrame([output_dict])
    html = df.round(decimals=2).T.to_html(
        classes=['table']
        )
    return html

def output_to_html_list(list):
    df = pd.DataFrame(list)
    html = df.round(decimals=2).to_html(
        classes=['table'],
        )
    return html


