class ConstraintBuilder:
    """adds name functionality to constraints in CVXPY"""

    def __init__(self):
        self.constraintList = []
        self.str2constr = {}

    def add_constraint(self, expr, str_):
        """add constraint with a string name"""
        self.constraintList.append(expr)
        self.str2constr[str_] = len(self.constraintList)-1

    def add_constraint_list(self, expr_list, str_list):
        if not len(expr_list) == len(str_list):
            raise ValueError('length of input lists should be equal')

        for i in range(len(expr_list)):
            self.add_constraint(expr=expr_list[i], str_=str_list[i])

    def get_constraint_list(self):
        """Get all constraints in the list"""
        return self.constraintList

    def get_constraint(self, str_):
        """takes the name of the constraints as argument,
        returns the corresponding constraint"""
        return self.constraintList[self.str2constr[str_]]

    def pop_constraint(self, str_):
        """takes 1 constraint name, removes that one from constraintList and str2constr"""
        self.constraintList.pop(self.str2constr[str_])
        index = self.str2constr[str_]
        del self.str2constr[str_]
        for key in self.str2constr.keys():
            if self.str2constr[key] > index:
                self.str2constr[key] += -1

    def pop_constraints(self, str_list):
        """remove a list of constraints at once."""
        for str_ in str_list:
            self.pop_constraint(str_=str_)
