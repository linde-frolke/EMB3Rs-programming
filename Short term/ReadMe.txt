Conventions:
1. In matrix notation, rows iterate over time, columns iterate over agents. So Pmin[1,5] is time 1 agent 5.
2. Index for an agent is denoted by "i", index for time is "t", index for node is "n", for pipe is "p"
3. Power injection by an agent or node is POSITIVE when it is a net generator.
4. Trade is positive when energy is sold
5. If a 3-dim array would be needed, cvxpy cannot handle this. The solution we use is a list of matrix variables.
    For example, trades are needed for each hour, and agent pair. The list Tnm contains a matrix variable for each t
    So Tnm[t][i,j] is the trade from i to j at time t.
