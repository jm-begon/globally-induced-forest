import os
from functools import partial

from .ruleset import AndCondition, LeqCondition, GrCondition
from .utils import fstr


class RulesetPrinter(object):
    def __init__(self, output_name=None, *variable_names, float_format="{:.2f}", ):
        self.output_name = output_name
        self.variable_names = variable_names
        self.float_format = float_format
        self.str_ = partial(fstr, format=float_format)

    def __repr__(self):
        return "{}({}, {}, *{})".format(self.__class__.__name__,
                                        repr(self.float_format),
                                        repr(self.output_name),
                                        repr(self.variable_names))

    def intercept_2_str(self, ruleset):
        return "intercept: {}".format(self.str_(ruleset.intercept))

    def and_cond_2_str(self, condition):
        return " and ".join(self.cond_2_str(cond) for cond in condition)

    def get_var_name(self, base_condition):
        index = base_condition.variable_index
        return self.variable_names[index] \
            if index < len(self.variable_names) \
            else "x_{}".format(index)


    def leq_cond_2_str(self, condition):
        val_str = self.float_format.format(condition.threshold)
        return "{} <= {}".format(self.get_var_name(condition), val_str)

    def gr_cond_2_str(self, condition):
        val_str = self.float_format.format(condition.threshold)
        return "{} > {}".format(self.get_var_name(condition), val_str)

    def cond_2_str(self, condition):
        if isinstance(condition, AndCondition):
            return self.and_cond_2_str(condition)
        if isinstance(condition, LeqCondition):
            return self.leq_cond_2_str(condition)
        if isinstance(condition, GrCondition):
            return self.gr_cond_2_str(condition)
        raise ValueError("Unknown condition class '{}' for '{}'"
                         "".format(condition.__class__.__name__,
                                   repr(condition)))

    def pred_2_str(self, prediction):
        pred_str = self.str_(prediction)
        if self.output_name is not None:
            pred_str = "{} = {}".format(self.output_name, pred_str)
        return pred_str



    def rule_2_str(self, rule):
        prop_str = self.float_format.format(rule.proportion)
        return "IF {} THEN {} ({})".format(self.cond_2_str(rule.condition),
                                           self.pred_2_str(rule.prediction),
                                           prop_str)

    def __call__(self, ruleset):
        lines = [self.intercept_2_str(ruleset)]
        for rule in ruleset:
            lines.append(self.rule_2_str(rule))

        return os.linesep.join(lines)