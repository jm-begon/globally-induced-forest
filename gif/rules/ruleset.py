import numpy as np

class Condition(object):
    pass

class BaseCondition(Condition):
    def __init__(self, variable_index, threshold):
        self.variable_index = variable_index
        self.threshold = threshold

    def __repr__(self):
        return "{}(variable_index={}, threshold={})" \
               "".format(self.__class__.__name__,
                         repr(self.variable_index),
                         repr(self.threshold))

    @property
    def variable_str(self):
        return "x_{}".format(self.variable_index)




class LeqCondition(BaseCondition):
    def __str__(self):
        return "{} <= {:.2f}".format(self.variable_str, self.threshold)

    def invert(self):
        return GrCondition(self.variable_index, self.threshold)

class GrCondition(BaseCondition):
    def __str__(self):
        return "{} > {:.2f}".format(self.variable_str, self.threshold)

    def invert(self):
        return LeqCondition(self.variable_index, self.threshold)

class AndCondition(Condition):
    @classmethod
    def flatten(cls, condition, accumulator=None):
        if isinstance(condition, AndCondition):
            for sub_cond in condition:
                cls.flatten(sub_cond, accumulator)
        else:
            accumulator.append(condition)

    @classmethod
    def by_flatting(cls, condition):
        if isinstance(condition, AndCondition):
            accumulator = []
            cls.flatten(condition, accumulator)
            return cls(*accumulator)

        return condition


    def __init__(self, *conditions):
        self.conditions = tuple(conditions)

    def __repr__(self):
        return "{}(*{})".format(self.__class__.__name__,
                                repr(self.conditions))

    def __str__(self):
        return " and ".join(str(cond) for cond in self.conditions)

    def __iter__(self):
        return iter(self.conditions)

    def __len__(self):
        return len(self.conditions)

class Rule(object):
    def __init__(self, condition, prediction):
        self.condition = condition
        self.prediction = prediction

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__,
                                   repr(self.condition), repr(self.prediction))

    def __str__(self):
        return "IF {} THEN {}".format(self.condition, self.prediction)

class RuleSet(object):
    @classmethod
    def tree_2_rules(cls, rules, tree, index=0, condition=None):
        if index < 0:
            return

        # If leaf (at least partially) prediction associated
        if tree.children_left[index] < 0 or tree.children_right[index] < 0:
            v = tree.value[index]
            if np.sum(v**2) > 1e-10:
                # Node actually holds a value
                rule = Rule(AndCondition.by_flatting(condition), v)
                rules.append(rule)

        # if internal node (at least partially) only part of condition
        this_condition = LeqCondition(tree.feature[index], tree.threshold[index])
        left_cond = this_condition
        if condition is not None:
            left_cond = AndCondition(condition, this_condition)
        cls.tree_2_rules(rules, tree, tree.children_left[index], left_cond)

        right_cond = this_condition.invert()
        if condition is not None:
            right_cond = AndCondition(condition, right_cond)
        cls.tree_2_rules(rules, tree, tree.children_right[index], right_cond)



    @classmethod
    def from_gif(cls, gif):
        rules = []
        intercept = gif.bias
        for tree in gif.estimators_:
            cls.tree_2_rules(rules, tree)

        return cls(intercept, rules)



    def __init__(self, intercept, rules):
        self.intercept_ = intercept
        self.rules_ = tuple(rules)

    def __repr__(self):
        return "{}(intercept={}, rules={})" \
               "".format(self.__class__.__name__,
                         repr(self.intercept_),
                         repr(self.rules_))

    @property
    def intercept(self):
        return self.intercept_


    def __iter__(self):
        return iter(self.rules_)

    def __len__(self):
        return len(self.rules_)
