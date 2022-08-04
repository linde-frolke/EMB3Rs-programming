from .market_functions.run_longterm_market import run_longterm_market
from .report.report_long_term import report_long_term

def main_longterm(in_var):

    longterm_results = run_longterm_market(in_var)
    report = report_long_term(longterm_results, in_var['user']['data_profile'])

    return {"results":longterm_results, "report":report}


