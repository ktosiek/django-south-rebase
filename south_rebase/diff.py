"""Classes and tools for diffing frozen models"""
from copy import deepcopy


class FieldDiff(object):
    empty = None

    def __init__(self, a, b):
        self.old = a
        self.new = b

    @property
    def is_deletion(self):
        return self.old and not self.new

    @property
    def is_addition(self):
        return self.new and not self.old

    def apply(self, base_field):
        if base_field is None:
            if not self.is_addition:
                raise Exception('Field not found')
            else:
                return self.new
        else:
            # TODO: check which parameters changed and only overwrite them
            return self.new

    @classmethod
    def diff(cls, a, b):
        return cls(a, b)

    @property
    def is_empty(self):
        return self.old == self.new

    def count_changes(self):
        plus = 0
        minus = 0
        if self.old:
            minus = 1
        if self.new:
            plus = 1
        return (plus, minus)

    def __repr__(self):
        return 'FieldDiff(%s, %s)' % (self.old, self.new)


class Diff(object):
    def __init__(self):
        self.changes = {}

    @property
    def is_empty(self):
        return all(lambda s: s.is_empty, self.changes.values())

    @property
    def is_addition(self):
        adds, dels = self.count_changes()
        return adds and not dels

    @property
    def is_deletion(self):
        adds, dels = self.count_changes()
        return dels and not adds

    def apply(self, base):
        new = deepcopy(base)
        for field, diff in self.changes.items():
            if field in base.keys():
                if not diff.is_deletion:
                    new[field] = diff.apply(new[field])
            elif diff.is_addition:
                new[field] = diff.apply(self.subdiff.empty)
        return new

    def count_changes(self):
        change_counts = zip(*(d.count_changes()
                              for d in self.changes.values()))
        return tuple(map(sum, change_counts))

    def __repr__(self):
        return '<%s: +%s/-%s>' % (
            (type(self).__name__,) + self.count_changes())


class ModelDiff(Diff):
    subdiff = FieldDiff
    empty = {}

    @classmethod
    def diff(cls, a, b):
        fields = set(a.keys()).union(b.keys())
        # TODO: Meta needs special treatment...
        fields.discard('Meta')

        diff = cls()
        for field in fields:
            field_diff = FieldDiff.diff(a.get(field), b.get(field))
            diff.changes[field] = field_diff

        return diff


class ModelsDiff(Diff):
    subdiff = ModelDiff
    empty = {}

    @classmethod
    def diff(cls, prev_models, models):
        diff = cls()
        models_for_diff = set(prev_models).union(models)
        for model in models_for_diff:
            model_diff = cls.subdiff.diff(prev_models.get(model),
                                          models.get(model))
            diff.changes[model] = model_diff
        return diff
