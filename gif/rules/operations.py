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


class SKTraversal(object):
    """
    For scikit-learn trees
    """
    def __init__(self, accumulator_factory, mode="prefix"):
        self.accumulator_factory = accumulator_factory
        self.mode = mode

    def forest_traversal(self, accumulator, forest):
        for t_idx, tree in enumerate(forest.estimators_):
            self.tree_traversal(accumulator, tree, t_idx)

    def tree_traversal(self, accumulator, tree, t_idx=0):
        if hasattr(tree, "tree_"):
            tree = tree.tree_
        self.tree_traversal_rec(accumulator, tree, t_idx)

    def tree_traversal_rec(self, accumulator, tree, t_idx=0, index=0, depth=0):
        if index < 0:
            return
        r_index = tree.children_right[index]
        l_index = tree.children_left[index]

        if self.mode == "prefix":
            accumulator(tree, t_idx, index, depth)
            self.tree_traversal_rec(accumulator, tree, t_idx, l_index, depth + 1)
            self.tree_traversal_rec(accumulator, tree, t_idx, r_index, depth + 1)
        elif self.mode == "infix":
            self.tree_traversal_rec(accumulator, tree, t_idx, l_index, depth + 1)
            accumulator(tree, t_idx, index, depth)
            self.tree_traversal_rec(accumulator, tree, t_idx, r_index, depth + 1)
        else:  # postfix
            self.tree_traversal_rec(accumulator, tree, t_idx, l_index, depth + 1)
            self.tree_traversal_rec(accumulator, tree, t_idx, r_index, depth + 1)
            accumulator(tree, t_idx, index, depth)

    def __call__(self, *somethings, **kwargs):
        accumulator = self.accumulator_factory(**kwargs)
        for something in somethings:
            if hasattr(something, "estimators_"):
                # sk forest
                self.forest_traversal(accumulator, something)
            elif hasattr(something, "tree_"):
                # sk tree
                self.tree_traversal(accumulator, something)
            else:
                self.tree_traversal_rec(accumulator, something)

        return accumulator


class Accumulator(object):
    def __call__(self, tree, t_idx, index, depth):
        pass

    def finalize(self):
        pass

class RuleCount(Accumulator):
    def __init__(self):
        self.count = 0

    def __call__(self, tree, t_idx, index, depth):
        r_index = tree.children_right[index]
        l_index = tree.children_left[index]

        if index > 0 and (l_index < 0 or r_index < 0):
            # Count only a (partial) leaf which is not a root
            self.count += 1

    def finalize(self):
        return self.count


class TotalComplexity(Accumulator):
    def __init__(self):
        self.complexity = 0

    def __call__(self, tree, t_idx, index, depth):
        r_index = tree.children_right[index]
        l_index = tree.children_left[index]

        if index > 0 and (l_index < 0 or r_index < 0):
            # Account only for a (partial) leaf which is not a root
            self.complexity += depth

    def finalize(self):
        return self.complexity


class BranchingHistogram(Accumulator):
    def __init__(self, max_depth=15):
        self.hist = np.zeros(max_depth, dtype=int)

    def __call__(self, tree, t_idx, index, depth):
        r_index = tree.children_right[index]
        l_index = tree.children_left[index]

        if index > 0 and (l_index < 0 or r_index < 0):
            # Account only for a (partial) leaf which is not a root
            self.hist[depth] += 1


    def rule_count(self):
        return self.hist[1:].sum()

    def complexity(self):
        return np.sum(np.arange(len(self.hist)) * self.hist)

    def finalize(self):
        return self.hist.copy()






