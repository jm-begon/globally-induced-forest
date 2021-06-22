import numpy as np

from .ruleset import AndCondition, LeqCondition, GrCondition

# To match Bénard, C., Biau, G., Veiga, S., & Scornet, E. (2021, March). Interpretable random forests via rule extraction. In International Conference on Artificial Intelligence and Statistics (pp. 937-945). PMLR.
class AlaSirusIntersector(object):
    def __init__(self, eps=1e-10):
        self.eps = eps

    def same_val(self, x1, x2):
        return np.sum((x1-x2)**2) < self.eps

    def __call__(self, rs1, rs2):
        commons = []
        for r1 in rs1:
            for r2 in rs2:
                if self.is_same(r1.condition, r2.condition):
                    commons.append(r1)
        return commons

    def is_same(self, cond1, cond2):
        if isinstance(cond1, AndCondition) and isinstance(cond2, AndCondition):
            if len(cond1) != len(cond2):
                return False
            for c1_i, c2_i in zip(cond1, cond2):
                if not self.is_same(c1_i, c2_i):
                    return False
            return True

        if isinstance(cond1, LeqCondition) and isinstance(cond2, LeqCondition):
            return cond1.variable_index == cond2.variable_index and \
                   self.same_val(cond1.threshold, cond2.threshold)

        if isinstance(cond1, GrCondition) and isinstance(cond2, GrCondition):
            return cond1.variable_index == cond2.variable_index and \
                   self.same_val(cond1.threshold, cond2.threshold)

        return False

def stability_ala_sirus(rs1, rs2, eps=1e-10):
    intersect = AlaSirusIntersector(eps)
    return 2.*len(intersect(rs1, rs2)) / float(len(rs1) + len(rs2))